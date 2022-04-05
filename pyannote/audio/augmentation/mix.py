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


from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_audiomentations import Mix
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict


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


class MixAugmentedSpeakerDiarization(MixSpeakerDiarization):
    """
    Create a new sample by mixing an original sample with another augmented random sample from the same batch.
    The applied augmentation is selected at random from a list of augmentations.

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    Parameters
    ----------
    augmentations: Union[BaseWaveformTransform, List[BaseWaveformTransform]]
        The list of augmentations to pick from to augment the samples that will be mixed.
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
        augmentations: Union[BaseWaveformTransform, List[BaseWaveformTransform]],
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
            max_num_speakers=max_num_speakers,
        )

        if isinstance(augmentations, BaseWaveformTransform):
            self.augmentations = [augmentations]
        else:
            self.augmentations = list(augmentations)

        for a in self.augmentations:
            if hasattr(a, "output_type") and a.output_type == "tensor":
                raise ValueError(
                    f"{self.__class__.__name__} only accepts augmentations with their output_type==dict"
                )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        uses_targets = targets is not None

        snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]

        background_samples = samples[idx].clone()
        if uses_targets:
            background_targets = targets[idx]

        # pick an augmentation for each sample
        augment_to_apply = (
            torch.rand(background_samples.shape[0]) * len(self.augmentations)
        ).int()

        # apply each augmentation to their assigned samples
        for augment_id in range(len(self.augmentations)):
            mask = augment_to_apply == augment_id
            result = self.augmentations[augment_id](
                samples=background_samples[mask],
                sample_rate=sample_rate,
                targets=background_targets[mask] if targets is not None else None,
                target_rate=target_rate,
            )
            background_samples[mask] = result.samples
            if uses_targets:
                background_targets[mask] = result.targets

        background_samples = Audio.rms_normalize(background_samples)
        background_rms = calculate_rms(samples) / (10 ** (snr.unsqueeze(dim=-1) / 20))

        mixed_samples = samples + background_rms.unsqueeze(-1) * background_samples

        if uses_targets:
            mixed_targets = self._mix_target(targets, background_targets, snr)
        else:
            mixed_targets = None

        return ObjectDict(
            samples=mixed_samples,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )
