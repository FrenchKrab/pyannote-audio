from typing import Sequence, Union
import torch
from torchmetrics.text import EditDistance


class TensorEditDistance(EditDistance):
    """Wrapper around torchmetrics.EditDistance to support tensors of integer/long values.
    """

    # TODO : move this to functional
    def _tensor_to_str(tensor: torch.Tensor) -> Union[str, Sequence[str]]:
        START_CHR = 65

        if tensor.ndim > 2:
            raise ValueError(f"Unsupported tensor shape : {tensor.shape}, only 1D and 2D tensors are supported.")
        elif tensor.ndim == 2:
            return [__class__._tensor_to_str(tensor[i]) for i in range(tensor.shape[0])]
        
        min_index = tensor.min().item()
        max_index = tensor.max().item()
        if max_index - min_index > 26:
            raise ValueError(f"Unsupported tensor values : {min_index} to {max_index}, only ranges of size 26 at most are supported for now.")

        return "".join([chr(int(i - min_index + START_CHR)) for i in tensor])

    def update(self, preds: Union[str, Sequence[str], torch.Tensor], target: Union[str, Sequence[str], torch.Tensor]) -> None:
        if isinstance(preds, torch.Tensor):
            preds = __class__._tensor_to_str(preds)
        if isinstance(target, torch.Tensor):
            target = __class__._tensor_to_str(target)

        super().update(preds, target)