from typing import Self, Sequence, Union
import torch
from torchmetrics.text import EditDistance
from zayrunner.utils.torch import unique_consecutive_padded


class TensorEditDistance(EditDistance):
    """Wrapper around torchmetrics.EditDistance to support tensors of integer/long values."""

    def __init__(self, blank_token, **kwargs):
        super().__init__(**kwargs)
        self.blank_token = blank_token

    # TODO : move this to functional
    def _tensor_to_str(self, tensor: torch.Tensor) -> Union[str, Sequence[str]]:
        START_CHR = 65

        if tensor.ndim > 3:
            raise ValueError(
                f"Unsupported tensor shape : {tensor.shape}, only 1D and 2D tensors are supported."
            )
        # dim==3: do argmax on last dim
        elif tensor.ndim == 3:
            if not tensor.is_floating_point():
                raise ValueError(
                    "3D tensor is not floating point. 3D tensors are only accepted when floating point with shape (batch_size, num_frames, num_classes)."
                )
            tensor = tensor.argmax(dim=-1)
            return self._tensor_to_str(tensor)
        # dim == 2: do unique_consecutive_padded to collapse consecutives
        elif tensor.ndim == 2:
            collapsed_tensor = unique_consecutive_padded(tensor, pad_value=-1)
            return [
                self._tensor_to_str(
                    collapsed_tensor[i][
                        (collapsed_tensor[i] != -1) & (collapsed_tensor[i] != self.blank_token)
                    ]
                )
                for i in range(tensor.shape[0])
            ]
        # dim == 1
        else:
            if tensor.numel() == 0:
                return ""
            min_index = tensor.min().item()
            max_index = tensor.max().item()
            if max_index - min_index > 26:
                raise ValueError(
                    f"Unsupported tensor values : {min_index} to {max_index}, only ranges of size 26 at most are supported for now."
                )

            return "".join([chr(int(i - min_index + START_CHR)) for i in tensor])

    def update(
        self,
        preds: Union[str, Sequence[str], torch.Tensor],
        target: Union[str, Sequence[str], torch.Tensor],
    ) -> None:
        if isinstance(preds, torch.Tensor):
            preds = self._tensor_to_str(preds)
        if isinstance(target, torch.Tensor):
            target = self._tensor_to_str(target)
        # print(f'-----\n{preds} \nvs\n {target}')
        super().update(preds, target)
