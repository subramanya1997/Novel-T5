#from core.utils import mytypes
#from typing import  Optional
import torch

def to_device(tt: torch.Tensor,
              device = 'cpu',
              non_blocking: bool = False) -> torch.Tensor:
    return tt.to(device, non_blocking=non_blocking)


def mktensor(data,
             dtype: torch.dtype = torch.float,
             device = 'cpu',
             requires_grad: bool = False,
             copy: bool = True) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
        is passed it is cast to  dtype, device and the requires_grad flag is
        set. This can copy data or make the operation in place.
    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
        copy: (bool): If false creates the tensor inplace else makes a copy
            (Default value = True)
    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data
    """
    tensor_factory = t if copy else t_
    return tensor_factory(
        data, dtype=dtype, device=device, requires_grad=requires_grad)