from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification.calibration_error import _binning_bucketize
from torchmetrics.utilities.data import dim_zero_cat


# TODO : should this stay, move it to its correct "functional" location
def _ce_compute(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Union[Tensor, int],
    bin_weighting: Literal["proportional", "uniform"] = "proportional",
    norm: str = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.
    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(
            0, 1, bin_boundaries + 1, dtype=torch.float, device=confidences.device
        )

    if norm not in {"l1", "l2", "max"}:
        raise ValueError(
            f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}"
        )

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(
            confidences, accuracies, bin_boundaries
        )

    if norm == "l1":
        if bin_weighting == "proportional":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        elif bin_weighting == "uniform":
            return torch.sum(torch.abs(acc_bin - conf_bin))
    if norm == "max":
        ce = torch.max(torch.abs(acc_bin - conf_bin))
    if norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
        if debias:
            # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
            # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                prop_bin * accuracies.size()[0] - 1
            )
            ce += torch.sum(
                torch.nan_to_num(debias_bins)
            )  # replace nans with zeros if nothing appeared in a bin
        return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    return ce

class BinaryCalibrationErrorV2(BinaryCalibrationError):
    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        bins_weighting: Literal["proportional", "uniform"] = "proportional",
        ignore_index: int = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_bins, norm, ignore_index, validate_args, **kwargs)
        if norm == "l2" and bins_weighting == "uniform":
            raise ValueError("L2 norm does not support uniform bins weighting yet.")

        self.bins_weighting = bins_weighting

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

class BinaryCalibrationErrorUniform(BinaryCalibrationErrorV2):
    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_bins, norm, bins_weighting="uniform", ignore_index=ignore_index, validate_args=validate_args, **kwargs
        )