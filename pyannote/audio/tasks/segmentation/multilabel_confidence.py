# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.database import Protocol
from pyannote.database.protocol import SegmentationProtocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, CalibrationError

from pyannote.audio.tasks import MultiLabelSegmentation
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks.segmentation.multilabel import Loggable


class MultiLabelSegmentationConfidence(MultiLabelSegmentation):
    """Generic multi-label segmentation

    Multi-label segmentation is the process of detecting temporal intervals
    when a specific audio class is active.

    Example use cases include speaker tracking, gender (male/female)
    classification, or audio event detection.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    classes : List[str], optional
        List of classes. Defaults to the list of classes available in the training set.
    duration : float, optional
        Chunks duration. Defaults to 2s.
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
    weight: str, optional
        When provided, use this key to as frame-wise weight in loss function.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    """
    def __init__(
        self,
        protocol: Protocol,
        budget: float,
        forced_exploration_ratio: float = 0.5,  # ratio of the batch that wont be affected by the confidence "cheating"
        lambda_multiplier: float = 0.99,    # how fast should we adjust lambda
        # normal args
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        metric_classwise: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        loggables: List[Loggable] = None,
    ):

        if not isinstance(protocol, SegmentationProtocol):
            raise ValueError(
                f"MultiLabelSegmentation task expects a SegmentationProtocol but you gave {type(protocol)}. "
            )

        super().__init__(
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
            metric_classwise=metric_classwise,
            balance=balance,
            weight=weight,
            classes=classes,
            loggables=loggables,
        )

        self.budget = budget
        self.lambda_multiplier = lambda_multiplier
        self.lmbda = 0.1
        self.forced_exploration_ratio = forced_exploration_ratio


    def training_step(self, batch, batch_idx: int):

        X = batch["X"]
        model_output = self.model(X)

        y_pred = model_output[:,:,:-1]
        y_true = batch["y"]
        y_confidence = model_output[:,:,-1:]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred_labelled = y_pred[mask]
        y_true_labelled = y_true[mask]
        y_confidence_labelled = y_confidence[mask]
        
        y_pred_cheated = y_confidence_labelled * y_pred_labelled + (1-y_confidence_labelled) * y_true_labelled 
        forced_exploration_count = int(self.forced_exploration_ratio * self.batch_size)
        y_pred_cheated[:forced_exploration_count] = y_pred_labelled[:forced_exploration_count]

        loss_l = F.binary_cross_entropy(y_pred_cheated, y_true_labelled.type(torch.float))
        loss_c = -torch.log(y_confidence_labelled).mean()
        loss = loss_l + self.lmbda * loss_c

        self.model.log(
            f"{self.logging_prefix}TrainLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.model.log(
            f"{self.logging_prefix}TrainLossCheatedBCE",
            loss_l,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.model.log(
            f"{self.logging_prefix}TrainLossConfidence",
            loss_c,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.model.log(
            f"{self.logging_prefix}TrainLambda",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        if self.budget > loss_c:
            self.lmbda *= self.lambda_multiplier
        else:
            self.lmbda *= 1/self.lambda_multiplier

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):

        X = batch["X"]
        model_output = self.model(X)

        y_pred = model_output[:,:,:-1]
        y_true = batch["y"]
        y_confidence = model_output[:,:,-1:]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # TODO: compute metrics for each class separately

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred_labelled = y_pred[mask]
        y_true_labelled = y_true[mask]
        y_confidence_labelled = y_confidence[mask]
        
        y_pred_cheated = y_confidence_labelled * y_pred_labelled + (1-y_confidence_labelled) * y_true_labelled 

        loss_real_bce = F.binary_cross_entropy(y_pred_labelled, y_true_labelled.type(torch.float))
        loss_l = F.binary_cross_entropy(y_pred_cheated, y_true_labelled.type(torch.float))
        loss_c = -torch.log(y_confidence_labelled).mean()
        loss = loss_l + loss_c

        # log loggables
        loggable_data = {
            "X": X,
            "y_pred": y_pred_labelled,
            "y_true": y_true_labelled,
            "y_conf": y_confidence,
        }
        for loggable in self.loggables:
            if loggable.update_in == "val":
                loggable.update(loggable_data)
            if loggable.compute_on == "step":
                loggable.log(self.model.loggers, self.model.current_epoch, batch_idx)
                loggable.clear()

        # log metrics per class
        for class_id, class_name in enumerate(self.classes):
            mask: torch.Tensor = y_true[..., class_id] != -1
            if mask.sum() == 0:
                continue

            y_pred_labelled = y_pred[..., class_id][mask]
            y_true_labelled = y_true[..., class_id][mask]

            metric = self.model.validation_metric_classwise[class_name]
            metric(
                y_pred_labelled,
                y_true_labelled,
            )

            self.model.log_dict(
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # log val confidence
        if self.model.val_confidence_metric is not None:
            self.model.val_confidence_metric(
                y_confidence.reshape((-1,y_pred.shape[-1])).squeeze(),
                (y_true==y_pred.mul(2).int().float()).reshape((-1,y_pred.shape[-1])).squeeze()
            )
            self.model.log(
                f"{self.logging_prefix}BinaryCalibrationErrorPrediction",
                self.model.val_confidence_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.model.validation_metric(
            y_pred.reshape((-1,y_pred.shape[-1])).squeeze(),
            y_true.reshape((-1,y_pred.shape[-1])).squeeze()
        )

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.model.log(
            f"{self.logging_prefix}ValLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.model.log(
            f"{self.logging_prefix}ValLossCheatedBCE",
            loss_l,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.model.log(
            f"{self.logging_prefix}ValLossBCE",
            loss_real_bce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.model.log(
            f"{self.logging_prefix}ValLossConfidence",
            loss_c,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss_real_bce}


    def setup_validation_metric(self):
        super().setup_validation_metric()

        self.model.val_confidence_metric = CalibrationError("binary", norm="l1")

    @property
    def val_monitor(self):
        """Quantity (and direction) to monitor

        Useful for model checkpointing or early stopping.

        Returns
        -------
        monitor : str
            Name of quantity to monitor.
        mode : {'min', 'max}
            Minimize

        See also
        --------
        pytorch_lightning.callbacks.ModelCheckpoint
        pytorch_lightning.callbacks.EarlyStopping
        """

        return f"{self.logging_prefix}ValLossBCE", "min"
