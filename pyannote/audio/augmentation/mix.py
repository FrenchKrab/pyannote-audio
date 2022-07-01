# MIT License
#
# Copyright (c) 2022- CNRS
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


from typing import Optional

import torch
from torch import Tensor
from torch_audiomentations import Mix
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class MixSpeakerDiarization(Mix):
    """
    Create a new sample by mixing it with another random sample from the same batch

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    Parameters
    ----------
    min_snr_in_db : float, optional
        Defaults to 0.0
    max_snr_in_db : float, optional
        Defaults to 5.0
    max_num_speakers: int, optional
        Maximum number of speakers in mixtures.  Defaults to actual maximum number
        of speakers in each batch.
    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = True

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        max_num_speakers: int = None,
        output_type: str = "tensor",
    ):
        super().__init__(
            min_snr_in_db=min_snr_in_db,
            max_snr_in_db=max_snr_in_db,
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.max_num_speakers = max_num_speakers

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape
        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # count number of active speakers per sample
        num_speakers: torch.Tensor = torch.sum(torch.any(targets, dim=-2), dim=-1)
        max_num_speakers = self.max_num_speakers or torch.max(num_speakers)

        # randomize index of second sample, constrained by the fact that the
        # resulting mixture should have less than max_num_speakers
        self.transform_parameters["sample_idx"] = torch.arange(
            batch_size, dtype=torch.int64
        )
        for n in range(max_num_speakers + 1):

            # indices of samples with exactly n speakers
            samples_with_n_speakers = torch.where(num_speakers == n)[0]
            num_samples_with_n_speakers = len(samples_with_n_speakers)
            if num_samples_with_n_speakers == 0:
                continue

            # indices of candidate samples for mixing (i.e. samples that would)
            candidates = torch.where(num_speakers + n <= max_num_speakers)[0]
            num_candidates = len(candidates)
            if num_candidates == 0:
                continue

            # sample uniformly from candidate samples
            selected_candidates = candidates[
                torch.randint(
                    0,
                    num_candidates,
                    (num_samples_with_n_speakers,),
                    device=samples.device,
                )
            ]
            self.transform_parameters["sample_idx"][
                samples_with_n_speakers
            ] = selected_candidates


class StitchAugmentation(BaseWaveformTransform):
    """
    Stitch two random segments from the same batch
    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        max_num_speakers: int,
        embedder: PretrainedSpeakerEmbedding = None,
        fade_duration: float = 0.5,  # In seconds, how much time to transition a track from on->off or off->on
        cut_margin: float = 0.25,  # In seconds, how much "extra time" to allocate around the cutting point
        cut_bound_low: float = 0.0,  # Minimum time (% of duration) at which the cut can occur
        cut_bound_high: float = 1.0,  # Maximum time (% of duration) at which the cut can occur
        fade_power: float = 2,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.max_num_speakers = max_num_speakers
        self.embedder = embedder
        self.fade_duration = fade_duration
        self.cut_margin = cut_margin
        self.cut_bound_low = cut_bound_low
        self.cut_bound_high = cut_bound_high
        self.fade_power = fade_power

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape

        cut_location_percent = (
            torch.rand(1) * (self.cut_bound_high - self.cut_bound_low)
            + self.cut_bound_low
        ).item()
        self.transform_parameters["cut_location"] = cut_location_percent
        cut_location = int(cut_location_percent * num_samples)

        fadein_start = round(max(cut_location - self.cut_margin * sample_rate, 0))
        fadeout_stop = round(
            min(cut_location + self.cut_margin * sample_rate, num_samples)
        )

        # Compute fade in/out tensors and store them
        fadein, fadeout = create_crossfade_tensor(
            fadein_start,
            fadeout_stop,
            int(self.fade_duration * sample_rate),
            num_samples,
        )
        self.transform_parameters["fade_in"] = fadein
        self.transform_parameters["fade_out"] = fadeout

        if self.embedder is not None:
            embeddings = self.embedder(samples)
            similarity_mat = cos_sim(embeddings, embeddings)

        # count number of active speakers per sample
        num_speakers: torch.Tensor = torch.sum(torch.any(targets, dim=-2), dim=-1)
        max_num_speakers = self.max_num_speakers or torch.max(num_speakers)

        # randomize index of second sample, constrained by the fact that the
        # resulting mixture should have less than max_num_speakers
        self.transform_parameters["sample_idx"] = torch.arange(
            batch_size, dtype=torch.int64
        )
        for n in range(max_num_speakers + 1):
            # indices of samples with exactly n speakers
            samples_with_n_speakers = torch.where(num_speakers == n)[0]
            num_samples_with_n_speakers = len(samples_with_n_speakers)
            if num_samples_with_n_speakers == 0:
                continue

            # indices of candidate samples for mixing (i.e. samples that would)
            candidates = torch.where(num_speakers + n <= max_num_speakers)[0]
            num_candidates = len(candidates)
            if num_candidates == 0:
                continue

            if self.embedder is not None:
                # create a indexing array with all non candidates
                non_candidates_indices = torch.ones(batch_size, dtype=bool)
                non_candidates_indices[candidates] = False

                similarity_mat_n = similarity_mat[samples_with_n_speakers].clone()
                similarity_mat_n.t()[
                    non_candidates_indices
                ] = 1.0  # make non candidates have the worst possible value
                most_different = torch.argmin(similarity_mat_n, dim=1)
                selected_candidates = most_different
            else:
                selected_candidates = candidates[
                    torch.randint(
                        0,
                        num_candidates,
                        (num_samples_with_n_speakers,),
                        device=samples.device,
                    )
                ]

            self.transform_parameters["sample_idx"][
                samples_with_n_speakers
            ] = selected_candidates

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        # snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]
        fade_in = self.transform_parameters["fade_in"]
        fade_out = self.transform_parameters["fade_out"]

        cut_location_percent = self.transform_parameters["cut_location"]

        mixed_samples = samples * (fade_out**self.fade_power) + samples[idx] * (
            fade_in * self.fade_power
        )

        if targets is None:
            mixed_targets = None
        else:
            firsthalf_targets, _ = torch.sort(targets, descending=False)
            secondhalf_targets, _ = torch.sort(targets[idx], descending=True)

            batch_size, num_channels, num_frames, num_speakers = targets.shape

            cut_location = round(cut_location_percent * num_frames)

            mixed_targets = firsthalf_targets.clone()
            mixed_targets[:, :, cut_location:, :] = secondhalf_targets[
                :, :, cut_location:, :
            ]

        return ObjectDict(
            samples=mixed_samples,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )


def create_crossfade_tensor(
    fadein_start: int,
    fadeout_stop: int,
    fade_duration: int,
    num_samples: int,
):
    fade_in = torch.ones(num_samples)
    fade_out = torch.ones(num_samples)

    in_start = fadein_start
    out_stop = fadeout_stop

    in_start_end = min(in_start + fade_duration, num_samples)
    out_stop_begin = max(0, out_stop - fade_duration)

    fade_in[:in_start] = 0.0
    fade_in[in_start:in_start_end] = (torch.linspace(0.0, 1.0, fade_duration))[
        : in_start_end - in_start
    ]
    fade_out[out_stop_begin:out_stop] = (torch.linspace(1.0, 0.0, fade_duration))[
        : out_stop - out_stop_begin
    ]
    fade_out[out_stop:] = 0.0

    return fade_in, fade_out


# source:
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L23
# under Apache 2 licence (https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# source :
# https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
# under Apache 2 licence (https://github.com/zhaobozb/layout2im/blob/master/LICENSE)
def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out
