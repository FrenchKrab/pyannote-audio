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


import itertools
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union

import numpy as np
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
import torch
import torch.nn.functional as F
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import ClasswiseWrapper, Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class MulticlassSegmentation(SegmentationTaskMixin, Task):
    """Overlapped speech detection

    Overlapped speech detection is the task of detecting regions where at least
    two speakers are speaking at the same time.

    Here, it is addressed with the same approach as voice activity detection,
    except "speech" class is replaced by "overlap", where a frame is marked as
    "overlap" if two speakers or more are active.

    Note that data augmentation is used to increase the proporition of "overlap".
    This is achieved by generating chunks made out of the (weighted) sum of two
    random chunks.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
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
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
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
        use_powerset: bool = True,
        label_scope: Literal["global", "database", "file"] = "file",
    ):
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
        self.use_powerset = use_powerset
        self.label_scope = label_scope

        # NEEDS classes to be passed
        print(f"classes: {self.classes}")
        if self.use_powerset:
            powersetclasses = ["nothing"]
            for simult_class_count in range(1, len(self.classes) + 1):
                for combination in itertools.combinations(self.classes, simult_class_count):
                    powersetclasses.append("+".join(combination))
            self.classes = powersetclasses
            
            print(f"classes after powerset: {self.classes}")

    def setup(self):
        super().setup()

        self.specifications = Specifications(
            classes=self.classes,
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
        )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk for multi-class segmentation

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
        y is shaped (num_frames, 1):
            -  0/1/2/...: class #1, #2, etc
            - -1: unknown class

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
        # TODO: add support for -1s
        y = -np.zeros((self.model.example_output.num_frames, 1), dtype=np.int8)
        # y[:, self.annotated_classes[file_id]] = 0
        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[f"{self.label_scope}_label_idx"]
        ):
            y[start:end, 0] = label + 1
            # y[start:end, 0] = 1 # because chunk_annotations["global_label_idx"] is always -1 for some reason
            # TODO: add support for powerset

        sample["y"] = SlidingWindowFeature(y, self.model.example_output.frames)

        # print(f"----\n{y.shape=}\n{self.model.example_output=}\n{self.model.example_output.num_frames=}\n{self.model.example_output.frames=}\n{chunk_annotations['global_label_idx']=}\n{chunk_annotations=}\n{start_idx=}\n{end_idx}")

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def training_step(self, batch, batch_idx: int):
        X = batch["X"]
        y_pred = self.model(X)
        y_true = batch["y"].squeeze()

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = F.nll_loss(y_pred, y_true.type(torch.long))

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
        y_true = batch["y"].squeeze().type(torch.long)

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = F.nll_loss(y_pred, y_true)

        # log global metric (multilabel)
        self.model.validation_metric(
            y_pred.argmax(dim=-1),
            y_true,
        )
        self.model.log_dict(
            self.model.validation_metric,
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

    def default_metric(self) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return {
            f"{self.classes[1]}/F1": BinaryF1Score(),
            f"{self.classes[1]}/Accuracy": BinaryAccuracy(),
            f"{self.classes[1]}/Precision": BinaryPrecision(),
            f"{self.classes[1]}/Recall": BinaryRecall(),
            # "ClasswiseF1": ClasswiseWrapper(
            #     MulticlassF1Score(num_classes=len(self.classes), average=None),
            #     labels=self.classes,
            #     postfix="/F1",
            # ),
            # "ClasswiseAccuracy": ClasswiseWrapper(
            #     MulticlassAccuracy(num_classes=len(self.classes), average=None),
            #     labels=self.classes,
            #     postfix="/Accuracy",
            # ),
            # "ClasswisePrecision": ClasswiseWrapper(
            #     MulticlassPrecision(num_classes=len(self.classes), average=None),
            #     labels=self.classes,
            #     postfix="/Precision",
            # ),
            # "ClasswiseRecall": ClasswiseWrapper(
            #     MulticlassRecall(num_classes=len(self.classes), average=None),
            #     labels=self.classes,
            #     postfix="/Recall",
            # ),
        }

    @property
    def val_monitor(self):
        return "loss/val", "min"
