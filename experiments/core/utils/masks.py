import torch
from typing import cast, Callable, Optional, Tuple


def pad_mask(lengths: torch.Tensor,
             max_length: Optional[int] = None,
             device='cpu'):
    """lengths is a torch tensor
    """
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length).unsqueeze(0).to(device)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask