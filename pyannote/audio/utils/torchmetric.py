# The MIT License (MIT)
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import numpy as np
import torch
from torchmetrics import Metric

from pyannote.audio.utils.metric import iterate_on_sfw
from pyannote.audio.utils.permutation import permutate
from pyannote.core import Annotation, SlidingWindowFeature, Timeline


class DiscreteDiarizationErrorRate(Metric):
    """Compute diarization error rate on discretized annotations with torchmetrics"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("confusion", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) != 3 or len(target.shape) != 3:
            msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
            raise ValueError(msg)

        batch_size, num_samples, num_classes_1 = target.shape
        batch_size_, num_samples_, num_classes_2 = preds.shape
        if (
            batch_size != batch_size_
            or num_samples != num_samples_
            or num_classes_1 != num_classes_2
        ):
            msg = f"Shape mismatch: {tuple(target.shape)} vs. {tuple(preds.shape)}."
            raise ValueError(msg)

        hypothesis, _ = permutate(target, preds)

        detection_error = torch.sum(hypothesis, 2) - torch.sum(target, 2)
        false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
        missed_detection = torch.maximum(
            -detection_error, torch.zeros_like(detection_error)
        )

        confusion = torch.sum((hypothesis != target) * hypothesis, 2) - false_alarm

        self.false_alarm += torch.sum(false_alarm)
        self.missed_detection += torch.sum(missed_detection)
        self.confusion += torch.sum(confusion)
        self.total += 1.0 * torch.sum(target)

    def update_from_sfw(
        self,
        reference: Annotation,
        hypothesis: SlidingWindowFeature,
        uem: Optional[Timeline] = None,
    ):
        def processing_fn(hypothesis, reference):
            # Prepare data
            ref_num_frames, ref_num_speakers = reference.shape

            if hypothesis.ndim != 2:
                raise NotImplementedError(
                    "Only (num_frames, num_speakers)-shaped hypothesis is supported."
                )

            hyp_num_frames, hyp_num_speakers = hypothesis.shape

            if ref_num_frames != hyp_num_frames:
                raise ValueError(
                    "reference and hypothesis must have the same number of frames."
                )

            if hyp_num_speakers > ref_num_speakers:
                reference = np.pad(
                    reference, ((0, 0), (0, hyp_num_speakers - ref_num_speakers))
                )
            elif ref_num_speakers > hyp_num_speakers:
                hypothesis = np.pad(
                    hypothesis, ((0, 0), (0, ref_num_speakers - hyp_num_speakers))
                )

            self.update(
                torch.tensor(hypothesis)[None, :, :],
                torch.tensor(reference)[None, :, :],
            )

        for _ in iterate_on_sfw(
            processing_fn, hypothesis=hypothesis, reference=reference, uem=uem
        ):
            pass

    def compute(self):
        return (self.false_alarm + self.missed_detection + self.confusion) / self.total
