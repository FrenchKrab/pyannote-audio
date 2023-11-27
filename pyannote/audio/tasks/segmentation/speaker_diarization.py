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
import math
import warnings
from collections import Counter
from typing import Callable, Dict, Literal, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from rich.progress import track
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.torchmetrics.text.TensorEditDistance import TensorEditDistance
from pyannote.audio.utils.loss import binary_cross_entropy, mse_loss, nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset

from zayrunner.utils.torch import unique_consecutive_padded

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)


class SpeakerDiarization(SegmentationTaskMixin, Task):
    """Speaker diarization

    Parameters
    ----------
    protocol : SpeakerDiarizationProtocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    max_speakers_per_chunk : int, optional
        Maximum number of speakers per chunk (must be at least 2).
        Defaults to estimating it from the training set.
    max_speakers_per_frame : int, optional
        Maximum number of (overlapping) speakers per frame.
        Setting this value to 1 or more enables `powerset multi-class` training.
        Default behavior is to use `multi-label` training.
    weigh_by_cardinality: bool, optional
        Weigh each powerset classes by the size of the corresponding speaker set.
        In other words, {0, 1} powerset class weight is 2x bigger than that of {0}
        or {1} powerset classes. Note that empty (non-speech) powerset class is
        assigned the same weight as mono-speaker classes. Defaults to False (i.e. use
        same weight for every class). Has no effect with `multi-label` training.
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
        When provided, use this key as frame-wise weight in loss function.
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
    vad_loss : {"bce", "mse"}, optional
        Add voice activity detection loss.
        Cannot be used in conjunction with `max_speakers_per_frame`.
    losses_data_filters: dict[str, Callable[[dict, dict], torch.Tensor]], optional
        Maps losses ('seg', 'ctc' or 'vad') to a data filter function.
        Only batch elements selected by the returned boolean torch filter mask
        will be used to compute the corresponding loss.
        Defaults to using all batch elements for all losses.
    losses_weights: float, optional
        Maps losses ('seg', 'ctc' or 'vad') to a weight.
        Defaults to 1.0 for all losses.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).

    References
    ----------
    Hervé Bredin and Antoine Laurent
    "End-To-End Speaker Segmentation for Overlap-Aware Resegmentation."
    Proc. Interspeech 2021

    Zhihao Du, Shiliang Zhang, Siqi Zheng, and Zhijie Yan
    "Speaker Embedding-aware Neural Diarization: an Efficient Framework for Overlapping
    Speech Diarization in Meeting Scenarios"
    https://arxiv.org/abs/2203.09767

    """

    DEFAULT_LOSSES_WEIGHTS = lambda _: {
        "seg": 1.0,
        "vad": 0.0,
        "ctc": 0.0,
        "antiblank": 0.0,
    }

    def __init__(
        self,
        protocol: SpeakerDiarizationProtocol,
        duration: float = 2.0,
        max_speakers_per_chunk: int = None,
        max_speakers_per_frame: int = None,
        weigh_by_cardinality: bool = False,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Sequence[Text] = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        vad_loss: Literal["bce", "mse"] = None,
        losses_data_filters: dict[str, Callable[[dict, dict], torch.Tensor]] = None,
        losses_weights: float = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        max_num_speakers: int = None,  # deprecated in favor of `max_speakers_per_chunk``
        loss: Literal["bce", "mse"] = None,  # deprecated
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

        if not isinstance(protocol, SpeakerDiarizationProtocol):
            raise ValueError(
                "SpeakerDiarization task requires a SpeakerDiarizationProtocol."
            )

        # deprecation warnings
        if max_speakers_per_chunk is None and max_num_speakers is not None:
            max_speakers_per_chunk = max_num_speakers
            warnings.warn(
                "`max_num_speakers` has been deprecated in favor of `max_speakers_per_chunk`."
            )
        if loss is not None:
            warnings.warn("`loss` has been deprecated and has no effect.")

        # initialize the dictionary of losses weights
        self.losses_weights = self.DEFAULT_LOSSES_WEIGHTS().copy()
        if losses_weights is not None:
            if len(set(losses_weights.keys()) - set(self.losses_weights.keys())) > 0:
                raise ValueError(
                    f"Unknown loss name. Valid loss names are {list(self.losses_weights.keys())}."
                )
            self.losses_weights.update(losses_weights)
        

        # initialize the dictionary of data filters
        self.losses_data_filters: dict[str, Callable[[dict, dict], torch.Tensor]] = {
            k: __class__._identity_data_filter for k in self.losses_weights.keys()
        }
        if losses_data_filters is not None:
            for k, v in losses_data_filters.items():
                if k not in self.losses_data_filters:
                    raise ValueError(
                        f"Unknown loss name {k}. Valid loss names are {list(self.losses_data_filters.keys())}."
                    )
                if isinstance(v, Callable):
                    self.losses_data_filters[k] = v
                else:
                    raise ValueError(
                        f"Loss data filter for loss {k} must be a callable."
                    )

        # parameter validation
        if max_speakers_per_frame is not None:
            if max_speakers_per_frame < 1:
                raise ValueError(
                    f"`max_speakers_per_frame` must be 1 or more (you used {max_speakers_per_frame})."
                )
            if vad_loss is not None and not self.loss_enabled("ctc"):
                raise ValueError(
                    "`vad_loss` cannot be used jointly with `max_speakers_per_frame`"
                )
        if self.loss_enabled("ctc") and max_speakers_per_frame is None:
            raise ValueError("TEMP: CTC must be used with powerset encoding for now.")

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.weigh_by_cardinality = weigh_by_cardinality
        self.balance = balance
        self.weight = weight
        self.vad_loss = vad_loss

    def _identity_data_filter(
        batch: dict, metadata_unique_values: dict
    ) -> torch.Tensor:
        return torch.ones(
            batch["y"].shape[0], dtype=torch.bool, device=batch["y"].device
        )

    def loss_enabled(self, loss_name: Literal["seg", "ctc", "vad"]):
        return self.losses_weights[loss_name] > 0.0

    def setup(self):
        super().setup()

        # estimate maximum number of speakers per chunk when not provided
        if self.max_speakers_per_chunk is None:
            training = self.metadata["subset"] == Subsets.index("train")

            num_unique_speakers = []
            progress_description = f"Estimating maximum number of speakers per {self.duration:g}s chunk in the training set"
            for file_id in track(
                np.where(training)[0], description=progress_description
            ):
                annotations = self.annotations[
                    np.where(self.annotations["file_id"] == file_id)[0]
                ]
                annotated_regions = self.annotated_regions[
                    np.where(self.annotated_regions["file_id"] == file_id)[0]
                ]
                for region in annotated_regions:
                    # find annotations within current region
                    region_start = region["start"]
                    region_end = region["end"]
                    region_annotations = annotations[
                        np.where(
                            (annotations["start"] >= region_start)
                            * (annotations["end"] <= region_end)
                        )[0]
                    ]

                    for window_start in np.arange(
                        region_start, region_end - self.duration, 0.25 * self.duration
                    ):
                        window_end = window_start + self.duration
                        window_annotations = region_annotations[
                            np.where(
                                (region_annotations["start"] <= window_end)
                                * (region_annotations["end"] >= window_start)
                            )[0]
                        ]
                        num_unique_speakers.append(
                            len(np.unique(window_annotations["file_label_idx"]))
                        )

            # because there might a few outliers, estimate the upper bound for the
            # number of speakers as the 97th percentile

            num_speakers, counts = zip(*list(Counter(num_unique_speakers).items()))
            num_speakers, counts = np.array(num_speakers), np.array(counts)

            sorting_indices = np.argsort(num_speakers)
            num_speakers = num_speakers[sorting_indices]
            counts = counts[sorting_indices]

            ratios = np.cumsum(counts) / np.sum(counts)

            for k, ratio in zip(num_speakers, ratios):
                if k == 0:
                    print(f"   - {ratio:7.2%} of all chunks contain no speech at all.")
                elif k == 1:
                    print(f"   - {ratio:7.2%} contain 1 speaker or less")
                else:
                    print(f"   - {ratio:7.2%} contain {k} speakers or less")

            self.max_speakers_per_chunk = max(
                2,
                num_speakers[np.where(ratios > 0.97)[0][0]],
            )

            print(
                f"Setting `max_speakers_per_chunk` to {self.max_speakers_per_chunk}. "
                f"You can override this value (or avoid this estimation step) by passing `max_speakers_per_chunk={self.max_speakers_per_chunk}` to the task constructor."
            )

        if (
            self.max_speakers_per_frame is not None
            and self.max_speakers_per_frame > self.max_speakers_per_chunk
        ):
            raise ValueError(
                f"`max_speakers_per_frame` ({self.max_speakers_per_frame}) must be smaller "
                f"than `max_speakers_per_chunk` ({self.max_speakers_per_chunk})"
            )

        # now that we know about the number of speakers upper bound
        # we can set task specifications
        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )

    def setup_loss_func(self):
        if self.specifications.powerset:
            self.model.powerset = Powerset(
                len(self.specifications.classes),
                self.specifications.powerset_max_classes,
            )

            self.model.metric_edit_distance = TensorEditDistance(
                blank_token=self.model.powerset.num_powerset_classes - 1
            )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk

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
            - `y`: target as a SlidingWindowFeature instance where y.labels is
                   in meta.scope space.
            - `meta`:
                - `scope`: target scope (0: file, 1: database, 2: global)
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        # get label scope
        label_scope = Scopes[self.metadata[file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        #
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

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        y = np.zeros((self.model.example_output.num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start:end, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.example_output.frames, labels=labels
        )

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def collate_y(self, batch) -> torch.Tensor:
        """

        Parameters
        ----------
        batch : list
            List of samples to collate.
            "y" field is expected to be a SlidingWindowFeature.

        Returns
        -------
        y : torch.Tensor
            Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
            If one chunk has more than `self.max_speakers_per_chunk` speakers, we keep
            the max_speakers_per_chunk most talkative ones. If it has less, we pad with
            zeros (artificial inactive speakers).
        """

        collated_y = []
        for b in batch:
            y = b["y"].data
            num_speakers = len(b["y"].labels)
            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y, axis=0), axis=0)
                # keep only the most talkative speakers
                y = y[:, indices[: self.max_speakers_per_chunk]]

                # TODO: we should also sort the speaker labels in the same way

            elif num_speakers < self.max_speakers_per_chunk:
                # create inactive speakers by zero padding
                y = np.pad(
                    y,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            else:
                # we have exactly the right number of speakers
                pass

            collated_y.append(y)

        return torch.from_numpy(np.stack(collated_y))

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
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

        if self.specifications.powerset:
            # `clamp_min` is needed to set non-speech weight to 1.
            class_weight = (
                torch.clamp_min(self.model.powerset.cardinality, 1.0)
                if self.weigh_by_cardinality
                else None
            )
            seg_loss = nll_loss(
                permutated_prediction,
                torch.argmax(target, dim=-1),
                class_weight=class_weight,
                weight=weight,
            )
        else:
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
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

    def _vad_loss_alt(
        self, prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Alternative vad loss for powerset prediction with blank token as last class

        Parameters
        ----------
        prediction : torch.Tensor
            Powerset prediction, where 0=nonspeech and -1=blank token
        target : torch.Tensor
            Multilabel target
        weight : torch.Tensor
            Weights

        Returns
        -------
        torch.Tensor
            VAD loss
        """
        prediction_exp = prediction.exp()
        vad_prediction = torch.concat(
            [
                prediction_exp[..., 0:1],
                torch.sum(prediction_exp[..., 1:-1], dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        # (batch_size, num_frames, 2) where last dim = (non-speech, speech)

        vad_target, _ = torch.max(target.float(), dim=-1, keepdim=False)
        vad_target = torch.cat(
            [
                (vad_target[..., None] - 1).abs(),
                vad_target[..., None],
            ], dim=-1
        )
        # (batch_size, num_frames, 2) where last dim = (non-speech, speech)

        return torch.nn.functional.binary_cross_entropy(
            vad_prediction,
            vad_target,
            weight=weight,
        )

    def _ctc_loss(
        self,
        prediction: torch.Tensor,
        target_multilabel: torch.Tensor,
        blank_id: int,
    ):
        ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id)

        # find best permutation for CTC, because we wouldn't be able to use permutate
        # with pure CTC annotations.
        best_ctc = torch.ones(size=(1,), device=target_multilabel.device) * torch.inf
        for permutation in itertools.permutations(
            range(target_multilabel.shape[-1]), target_multilabel.shape[-1]
        ):
            ps: Powerset = self.model.powerset

            ctc_perm_target_ml = target_multilabel[..., permutation]
            ctc_perm_target_ps = ps.to_powerset(ctc_perm_target_ml.float())
            ctc_perm_targets_cids = ctc_perm_target_ps.argmax(dim=-1)

            # (batch_size, num_frames) both
            collapsed_target, ct_indices = unique_consecutive_padded(
                ctc_perm_targets_cids, return_indices=True
            )
            # (batch_size)
            target_lengths, _ = torch.max(ct_indices, dim=-1)
            target_lengths += 1

            ctc = ctc_loss_fn(
                prediction.permute(1, 0, 2),
                collapsed_target,
                torch.ones((prediction.shape[0],), dtype=torch.long)
                * prediction.shape[1],
                target_lengths,
            )
            best_ctc = torch.min(best_ctc, ctc)
        # normalize ctc loss by prediction sequence length
        return best_ctc/prediction.shape[1]

    def training_step(self, batch, batch_idx: int):
        """Compute permutation-invariant segmentation loss

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
        keep: torch.Tensor = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        waveform = waveform[keep]

        # corner case
        if not keep.any():
            return None

        # forward pass
        prediction = self.model(waveform)
        batch_size, num_frames, _ = prediction.shape
        # (batch_size, num_frames, num_classes)

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

        seg_fmask = self.losses_data_filters["seg"](batch, self.metadata_unique_values)
        seg_fcount = seg_fmask.sum()

        vad_fmask = self.losses_data_filters["vad"](batch, self.metadata_unique_values)
        vad_fcount = vad_fmask.sum()

        ctc_fmask = self.losses_data_filters["ctc"](batch, self.metadata_unique_values)
        ctc_fcount = ctc_fmask.sum()

        antiblank_fmask = self.losses_data_filters["antiblank"](
            batch, self.metadata_unique_values
        )
        antiblank_fcount = antiblank_fmask.sum()

        seg_loss = 0.0
        ctc_loss = 0.0
        vad_loss = 0.0
        antiblank_loss = 0.0


        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            perm_target_ml, _ = permutate(multilabel, target)
            perm_target_ps = self.model.powerset.to_powerset(perm_target_ml.float())

            # If we use CTC, modify our encoding so that the last label is the blank token
            blank_id = self.model.powerset.num_powerset_classes - 1
            if self.loss_enabled("ctc"):
                labelled_blank = perm_target_ps[..., blank_id] == 1
                perm_target_ps[..., blank_id][labelled_blank] = 0
                perm_target_ps[..., blank_id-1][labelled_blank] = 1

            if self.loss_enabled("ctc") and ctc_fcount > 0:
                ctc_loss = self._ctc_loss(
                    prediction[ctc_fmask],
                    target[ctc_fmask],
                    blank_id=blank_id,
                )

            # compute segmentation loss
            if self.loss_enabled("seg") and seg_fcount > 0:
                seg_loss = self.segmentation_loss(
                    prediction[seg_fmask],
                    perm_target_ps[seg_fmask],
                    weight=weight[seg_fmask],
                )

            # Compute VAD loss
            if self.loss_enabled("vad") and vad_fcount > 0:
                vad_loss = self._vad_loss_alt(
                    prediction[vad_fmask],
                    target[vad_fmask],
                    weight=weight[vad_fmask],
                )

            # Compute "anti blank token" loss
            if self.loss_enabled("antiblank") and antiblank_fcount > 0:
                antiblank_loss = torch.nn.functional.binary_cross_entropy(
                    prediction[antiblank_fmask][...,-1].exp(),
                    torch.zeros_like(prediction[antiblank_fmask][...,-1]),
                    weight=weight[antiblank_fmask].squeeze(-1),
                )


        # Multilabel
        else:
            if self.loss_enabled("seg") and seg_fcount > 0:
                permutated_prediction, _ = permutate(target, prediction)
                seg_loss = self.segmentation_loss(
                    permutated_prediction[seg_fmask],
                    target[seg_fmask],
                    weight=weight[seg_fmask],
                )

            if self.loss_enabled("vad") and vad_fcount > 0:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

        if seg_loss != 0.0:
            self.model.log(
                "loss/train/segmentation",
                seg_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if ctc_loss != 0.0:
            self.model.log(
                "loss/train/ctc",
                ctc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if vad_loss != 0.0:
            self.model.log(
                "loss/train/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if antiblank_loss != 0.0:
            self.model.log(
                "loss/train/antiblank",
                antiblank_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        loss = (
            self.losses_weights["seg"] * seg_loss
            + self.losses_weights["vad"] * vad_loss
            + self.losses_weights["ctc"] * ctc_loss
            + self.losses_weights["antiblank"] * antiblank_loss
        )

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            raise Exception("NaN loss")
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

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components"""

        if self.specifications.powerset:
            return {
                "DiarizationErrorRate": DiarizationErrorRate(0.5),
                "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
                "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
                "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
            }

        return {
            "DiarizationErrorRate": OptimalDiarizationErrorRate(),
            "DiarizationErrorRate/Threshold": OptimalDiarizationErrorRateThreshold(),
            "DiarizationErrorRate/Confusion": OptimalSpeakerConfusionRate(),
            "DiarizationErrorRate/Miss": OptimalMissedDetectionRate(),
            "DiarizationErrorRate/FalseAlarm": OptimalFalseAlarmRate(),
        }

    # TODO: no need to compute gradient in this method
    def validation_step(self, batch, batch_idx: int):
        """Compute validation loss and metric

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # TODO: should we handle validation samples with too many speakers
        # waveform = waveform[keep]
        # target = target[keep]

        # forward pass
        prediction = self.model(waveform)
        batch_size, num_frames, _ = prediction.shape

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

        seg_loss = 0.0
        vad_loss = 0.0
        ctc_loss = 0.0
        antiblank_loss = 0.0
        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            perm_target_ml, _ = permutate(multilabel, target)
            perm_target_ps = self.model.powerset.to_powerset(perm_target_ml.float())

            # If we use CTC, modify our encoding so that the last label is the blank token
            blank_id = self.model.powerset.num_powerset_classes - 1
            if self.loss_enabled("ctc"):
                labelled_blank = perm_target_ps[..., blank_id] == 1
                perm_target_ps[...,blank_id][labelled_blank] = 0
                perm_target_ps[...,blank_id-1][labelled_blank] = 1

                self.model.log(
                    "CTC/blankprob",
                    prediction[..., blank_id].exp().mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

            # Compute CTC loss
            ctc_loss = self._ctc_loss(
                prediction,
                target,
                blank_id=blank_id,
            )

            # compute segmentation loss
            seg_loss = self.segmentation_loss(
                prediction,
                perm_target_ps,
                weight=weight,
            )

            # Compute VAD loss
            vad_loss = self._vad_loss_alt(
                prediction,
                target,
                weight=weight,
            )


            # Compute "anti blank token" loss
            if self.loss_enabled("antiblank"):
                antiblank_loss = torch.nn.functional.binary_cross_entropy(
                    prediction[...,-1:].exp(),
                    torch.zeros_like(prediction[..., -1:]),
                    weight=weight,
                )
        else:
            permutated_prediction, _ = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )

            vad_loss = self.voice_activity_detection_loss(
                permutated_prediction, target, weight=weight
            )

        if vad_loss != 0.0:
            self.model.log(
                "loss/val/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if seg_loss != 0.0:
            self.model.log(
                "loss/val/segmentation",
                seg_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if antiblank_loss != 0.0:
            self.model.log(
                "loss/val/antiblank",
                antiblank_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if ctc_loss != 0.0:
            self.model.log(
                "loss/val/ctc",
                ctc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = (
            self.losses_weights["seg"] * seg_loss
            + self.losses_weights["vad"] * vad_loss
            + self.losses_weights["ctc"] * ctc_loss
            + self.losses_weights["antiblank"] * antiblank_loss
        )

        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # Compute validation metric (& edit distance if relevant)
        if self.specifications.powerset:
            self.model.validation_metric(
                torch.transpose(
                    multilabel[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
            )

            # Compute edit distance
            self.model.metric_edit_distance(
                prediction.argmax(dim=-1),
                perm_target_ps.argmax(dim=-1),
            )
            self.model.log(
                "CTC/edit_distance",
                self.model.metric_edit_distance,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.model.validation_metric(
                torch.transpose(
                    prediction[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
            )

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # log first batch visualization every 2^n epochs.
        if (
            self.model.current_epoch == 0
            or math.log2(self.model.current_epoch) % 1 > 0
            or batch_idx > 0
        ):
            return

        # visualize first 9 validation samples of first batch in Tensorboard/MLflow

        if self.specifications.powerset:
            y = perm_target_ml.float().cpu().numpy()
            # y_pred = multilabel.cpu().numpy()
            y_pred = prediction.exp().cpu().numpy()
        else:
            y = target.float().cpu().numpy()
            y_pred = permutated_prediction.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.NaN
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):
            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot target
            ax_ref = axes[row_idx * 2 + 0, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot predictions
            ax_hyp = axes[row_idx * 2 + 1, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.axvspan(0, warm_up_left, color="k", alpha=0.5, lw=0)
            ax_hyp.axvspan(
                num_frames - warm_up_right, num_frames, color="k", alpha=0.5, lw=0
            )
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

            if self.specifications.powerset:
                collapsed_y = unique_consecutive_padded(
                    perm_target_ps[sample_idx].argmax(dim=-1)
                )
                collapsed_y = collapsed_y[collapsed_y != -1]

                collapsed_y_pred = unique_consecutive_padded(
                    prediction[sample_idx].argmax(dim=-1)
                )
                collapsed_y_pred = collapsed_y_pred[
                    (collapsed_y_pred != -1)
                    & (collapsed_y_pred != self.model.powerset.num_powerset_classes - 1)
                ]

                str_y = ".".join([str(x) for x in collapsed_y.tolist()])
                str_y_pred = ".".join([str(x) for x in collapsed_y_pred.tolist()])

                ax_ref.text(0.1, 0.8, str_y)
                ax_hyp.text(0.1, 0.2, str_y_pred)

        plt.tight_layout()

        for logger in self.model.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("samples", fig, self.model.current_epoch)
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"samples_epoch{self.model.current_epoch}.png",
                )

        plt.close(fig)

    @property
    def val_monitor(self):
        return (
            ("loss/val", "min")
        )


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

        def progress_hook(completed: int = None, total: int = None):
            progress.update(file_task, completed=completed / total)

        inference = Inference(model, device=device)

        for file in files:
            progress.update(file_task, description=file["uri"])
            reference = file["annotation"]
            hypothesis = binarize(inference(file, hook=progress_hook))
            uem = file["annotated"]
            _ = metric(reference, hypothesis, uem=uem)
            progress.advance(main_task)

    _ = metric.report(display=True)


if __name__ == "__main__":
    import typer

    typer.run(main)
