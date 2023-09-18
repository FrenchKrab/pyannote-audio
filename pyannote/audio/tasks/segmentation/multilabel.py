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

from functools import cached_property
from typing import Dict, List, Optional, Sequence, Text, Tuple, Union
from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
from pyannote.database.protocol import SegmentationProtocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Accuracy, F1Score, Metric, MetricCollection, Precision, Recall

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class MultiLabelSegmentation(SegmentationTaskMixin, Task):
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
    balance: Sequence[Text], optional
        When provided, training samples are sampled uniformly with respect to these keys.
        For instance, setting `balance` to ["database","subset"] will make sure that each
        database & subset combination will be equally represented in the training samples.
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
        Multilabel validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Make sure the metric's ignore_index==-1 if your data contains un-annotated frames.
        Defaults to F1, Precision, Recall & Accuracy in macro mode.
    metric_classwise: Union[Metric, Sequence[Metric], Dict[str, Metric]], optional
        Validation metric(s) to compute for each class (binary). Can be anything supported by torchmetrics.MetricCollection.
        No need for ignore_index=-1 here, as the metric is computed only on the labelled frames.
        Defaults to F1, Precision, Recall & Accuracy.
    """

    def __init__(
        self,
        protocol: Protocol,
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Sequence[Text] = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        metric_classwise: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
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
        )

        self.balance = balance
        self.weight = weight
        self.classes = classes
        self._metric_classwise = metric_classwise

        # task specification depends on the data: we do not know in advance which
        # classes should be detected. therefore, we postpone the definition of
        # specifications to setup()

    def setup(self):
        super().setup()

        self.specifications = Specifications(
            classes=self.classes,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
        )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk for multi-label segmentation

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target (see Notes below)
            - `meta`:
                - `database`: database index
                - `file`: file index

        Notes
        -----
        y is a trinary matrix with shape (num_frames, num_classes):
            -  0: class is inactive
            -  1: class is active
            - -1: we have no idea

        """

        file = self.get_file(file_id)

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)
        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        start_idx = np.floor(start / self.model.example_output.frames.step).astype(int)
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / self.model.example_output.frames.step).astype(int)

        # frame-level targets (-1 for un-annotated classes)
        y = -np.ones(
            (self.model.example_output.num_frames, len(self.classes)), dtype=np.int8
        )
        y[:, self.annotated_classes[file_id]] = 0
        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations["global_label_idx"]
        ):
            y[start:end, label] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.example_output.frames, labels=self.classes
        )

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def training_step(self, batch, batch_idx: int):
        X = batch["X"]
        y_pred = self.model(X)
        y_true = batch["y"]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = F.binary_cross_entropy(y_pred, y_true.type(torch.float))

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        X = batch["X"]
        y_pred = self.model(X)
        y_true = batch["y"]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred_labelled = y_pred[mask]
        y_true_labelled = y_true[mask]
        loss = F.binary_cross_entropy(
            y_pred_labelled, y_true_labelled.type(torch.float)
        )

        # log global metric (multilabel)
        self.model.validation_metric(
            y_pred.reshape((-1, y_pred.shape[-1])),
            y_true.reshape((-1, y_true.shape[-1])),
        )
        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # log metrics per class (binary)
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

        # log losses
        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        class_count = len(self.classes)
        if class_count > 1:  # multilabel
            return {
                "F1": F1Score(
                    task="multilabel",
                    num_labels=class_count,
                    ignore_index=-1,
                    average="macro",
                ),
                "Precision": Precision(
                    task="multilabel",
                    num_labels=class_count,
                    ignore_index=-1,
                    average="macro",
                ),
                "Recall": Recall(
                    task="multilabel",
                    num_labels=class_count,
                    ignore_index=-1,
                    average="macro",
                ),
                "Accuracy": Accuracy(
                    task="multilabel",
                    num_labels=class_count,
                    ignore_index=-1,
                    average="macro",
                ),
            }
        else:
            # Binary classification, this case is handled by the per-class metric, see 'default_metric_per_class'/'metric_classwise'
            return []

    def default_metric_classwise(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return {
            "F1": F1Score(task="binary"),
            "Precision": Precision(task="binary"),
            "Recall": Recall(task="binary"),
            "Accuracy": Accuracy(task="binary"),
        }

    @cached_property
    def metric_classwise(self) -> MetricCollection:
        if self._metric_classwise is None:
            self._metric_classwise = self.default_metric_classwise()

        return MetricCollection(self._metric_classwise)

    def setup_validation_metric(self):
        # setup global/multilabel validation metric
        super().setup_validation_metric()

        # and then setup validation metric per class / classwise metrics
        metric = self.metric_classwise
        if metric is None:
            return

        self.model.validation_metric_classwise = torch.nn.ModuleDict().to(
            self.model.device
        )
        for class_name in self.classes:
            self.model.validation_metric_classwise[class_name] = metric.clone(
                prefix=f"{class_name}/"
            )

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

        return "loss/val", "min"
