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

from ast import Pow
from collections import Counter
import math
from typing import Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
import itertools
from pyannote.core import SlidingWindow
from pyannote.database import Protocol
from torch.nn.modules import padding
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.torchmetrics import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
)
from pyannote.audio.utils.loss import binary_cross_entropy, mse_loss, nll_loss
from pyannote.audio.utils.permutation import permutate


class SegmentationPowerset(SegmentationTaskMixin, Task):
    """Speaker segmentation using a power set encoding of active speakers.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    max_num_speakers : int
        Maximum number of speakers per chunk (must be at least 2).
    max_simultaneous_speakers : int
        Maximum number of simultaneous speakers per frame.
        Limits the maximum cardinality of the subsets of all speakers,
        setting it to max_num_speakers result in the "true" power set encoding
        (= all possible speaker combinations have a class).
        Must be in [1, max_num_speakers].
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
    loss : {"bce", "mse"}, optional
        Permutation-invariant segmentation loss. Defaults to "bce".
    vad_loss : {"bce", "mse"}, optional
        Add voice activity detection loss.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).

    Reference
    ----------
    Hervé Bredin and Antoine Laurent
    "End-To-End Speaker Segmentation for Overlap-Aware Resegmentation."
    Proc. Interspeech 2021
    """

    def __init__(
        self,
        protocol: Protocol,
        max_num_speakers: int,
        max_simult_speakers: int,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        loss: Literal["bce", "mse", "nll"] = "nll",
        vad_loss: Literal["bce", "mse"] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
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

        self.max_num_speakers = max_num_speakers
        self.max_simult_speakers = max_simult_speakers

        self.balance = balance
        self.weight = weight

        if loss not in ["bce", "mse", "nll"]:
            raise ValueError("'loss' must be one of {'bce', 'mse', 'nll'}.")
        self.loss = loss
        self.vad_loss = vad_loss

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)

        # now that we know about the number of speakers upper bound
        # we can set task specifications
        self.specifications = Specifications(
            problem=Problem.POWERSET,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_num_speakers)],
            max_simult_speakers=self.max_simult_speakers,
            permutation_invariant=True,
        )

    def adapt_y(self, collated_y: torch.Tensor) -> torch.Tensor:
        """Get speaker diarization targets

        Parameters
        ----------
        collated_y : (batch_size, num_frames, num_speakers) tensor
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[b, f, s] = 1 if sth speaker is active at fth frame
                * one_hot_y[b, f, s] = 0 otherwise.

        Returns
        -------
        y : (batch_size, num_frames, max_num_speakers) tensor
            Same as collated_y, except we only keep ``max_num_speakers`` most
            talkative speakers (per sample).
        """

        batch_size, num_frames, num_speakers = collated_y.shape

        # maximum number of active speakers in a chunk
        max_num_speakers = torch.max(
            torch.sum(torch.sum(collated_y, dim=1) > 0.0, dim=1)
        )

        # sort speakers in descending talkativeness order
        indices = torch.argsort(torch.sum(collated_y, dim=1), dim=1, descending=True)

        # keep max_num_speakers most talkative speakers, for each chunk
        y = torch.zeros(
            (batch_size, num_frames, max_num_speakers), dtype=collated_y.dtype
        )
        for b, index in enumerate(indices):
            for k, i in zip(range(max_num_speakers), index):
                y[b, :, k] = collated_y[b, :, i.item()]

        return y

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes_mono) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        if self.loss == "bce":
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
            )

        elif self.loss == "mse":
            seg_loss = mse_loss(permutated_prediction, target.float(), weight=weight)

        elif self.loss == "nll":
            seg_loss = nll_loss(
                permutated_prediction, torch.argmax(target, dim=-1), weight=weight
            )

        return seg_loss

    def voice_activity_detection_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Voice activity detection loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        vad_loss : torch.Tensor
            Voice activity detection loss.
        """

        vad_prediction, _ = torch.max(permutated_prediction, dim=2, keepdim=True)
        # (batch_size, num_frames, 1)

        vad_target, _ = torch.max(target.float(), dim=2, keepdim=False)
        # (batch_size, num_frames)

        if self.vad_loss == "bce":
            loss = binary_cross_entropy(vad_prediction, vad_target, weight=weight)

        elif self.vad_loss == "mse":
            loss = mse_loss(vad_prediction, vad_target, weight=weight)

        return loss

    def training_step(self, batch, batch_idx: int):
        """Compute permutation-invariant binary cross-entropy

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # drop samples that contain too many speakers
        num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
        keep: torch.Tensor = num_speakers <= self.max_num_speakers
        target = target[keep]
        waveform = waveform[keep]

        # log effective batch size
        self.model.log(
            f"{self.logging_prefix}BatchSize",
            keep.sum(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            reduce_fx="mean",
        )

        # corner case
        if not keep.any():
            return {"loss": 0.0}

        # forward pass
        prediction = self.model(waveform)
        batch_size, num_frames, _ = prediction.shape
        # (batch_size, num_frames, num_classes)

        # find optimal permutation between the one hot of our multiclass-softmax and the multilabel target
        # and use it to permutate target
        one_hot_prediction = torch.nn.functional.one_hot(
            torch.argmax(prediction, dim=-1), self.specifications.powerset_class_count
        ).float()
        one_hot_prediction_multi = self.powerset_to_multilabel(one_hot_prediction)

        permutated_target, _ = permutate(one_hot_prediction_multi, target)
        permutated_target_mono = self.multilabel_to_powerset(permutated_target)

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        seg_loss = self.segmentation_loss(
            prediction, permutated_target_mono, weight=weight
        )

        self.model.log(
            f"{self.logging_prefix}TrainSegLoss",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            vad_loss = self.voice_activity_detection_loss(
                prediction, permutated_target_mono, weight=weight
            )

            self.model.log(
                f"{self.logging_prefix}TrainVADLoss",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        self.model.log(
            f"{self.logging_prefix}TrainLoss",
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
        """Returns diarization error rate and its components"""
        return [
            OptimalDiarizationErrorRate(),
            OptimalDiarizationErrorRateThreshold(),
            OptimalSpeakerConfusionRate(),
            OptimalMissedDetectionRate(),
            OptimalFalseAlarmRate(),
        ]

    def train__iter__(self):
        base_train_iter = super().train__iter__()
        # Only yield chunks with number of speakers <= max_num_speakers, so that we end up with batch target tensors
        # of speaker dimension size <= max_num_speakers.
        while True:
            chunk = next(base_train_iter)
            if len(chunk["y"].labels) <= self.max_num_speakers:
                yield chunk

    def setup_loss_func(self):
        # save our handy conversion tensor in the model (no need for persistence)
        conversion_tensor = Problem.build_powerset_to_multi_conversion_tensor(
            self.max_num_speakers, self.max_simult_speakers
        )
        self.model.register_buffer(
            "powerset_conversion_tensor", conversion_tensor, persistent=False
        )

        # call default setup
        super().setup_loss_func()

    @property
    def mono_to_multi_tensor(self):
        return self.model.powerset_conversion_tensor

    # Monolabel <-> multilabel problem conversion helpers (using cached conversion tensor !)
    def multilabel_to_powerset(self, t: torch.Tensor) -> torch.Tensor:
        return self.specifications.multilabel_to_powerset(
            t,
            self.mono_to_multi_tensor,
        )

    def powerset_to_multilabel(self, t: torch.Tensor) -> torch.Tensor:
        return self.specifications.powerset_to_multilabel(
            t,
            self.mono_to_multi_tensor,
        )

    def convert_to_powerset_permutation(self, permutation: torch.Tensor):
        return Problem.get_powerset_permutation(
            permutation,
            self.max_num_speakers,
            self.max_simult_speakers,
            self.specifications.powerset_conversion_dict,
            self.specifications.powerset_conversion_dict_inv,
        )


def mono_nll_loss(target, preds):
    return torch.nn.functional.nll_loss(preds.float(), torch.argmax(target, dim=-1))


def mono_mse_loss(target, preds):
    return torch.nn.functional.mse_loss(preds, target)


def main(protocol: str, subset: str = "test", model: str = "pyannote/segmentation"):
    """Evaluate a segmentation model"""

    from pyannote.database import FileFinder, get_protocol
    from rich.progress import Progress

    from pyannote.audio import Inference
    from pyannote.audio.pipelines.utils import get_devices
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.utils.signal import binarize

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})
    files = list(getattr(protocol, subset)())

    with Progress() as progress:

        main_task = progress.add_task(protocol.name, total=len(files))
        file_task = progress.add_task("Processing", total=1.0)

        def progress_hook(completed: int, total: int):
            progress.update(file_task, completed=completed / total)

        inference = Inference(model, device=device, progress_hook=progress_hook)

        for file in files:
            progress.update(file_task, description=file["uri"])
            reference = file["annotation"]
            hypothesis = binarize(inference(file))
            uem = file["annotated"]
            _ = metric(reference, hypothesis, uem=uem)
            progress.advance(main_task)

    _ = metric.report(display=True)


if __name__ == "__main__":
    import typer

    typer.run(main)
