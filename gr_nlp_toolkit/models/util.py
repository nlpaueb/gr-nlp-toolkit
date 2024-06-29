import torch


def create_mask_from_length(length_tensor, mask_size):

    """
    Creates a binary mask based on length.

    Args:
        length_tensor (torch.Tensor): ND Tensor containing the lengths.
        mask_size (int): Integer specifying the mask size. Usually the largest length in the batch
    
    Return:
        torch.Tensor (N+1)D Int Tensor (..., mask_size) containing the binary mask.
    """

    mask = torch.arange(0, mask_size, dtype=torch.int, device=length_tensor.device)
    
    mask = mask.int().view([1] * (len(length_tensor.shape)) + [-1])

    return mask < length_tensor.int().unsqueeze(-1)


def get_device_name() -> Literal["mps", "cuda", "cpu"]:
    """
    Returns the name of the device where this module is running.

    This is a simple implementation that doesn't cover cases when more powerful GPUs are available 
    and not a primary device ('cuda:0') or MPS device is available but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    Returns:
        Literal["mps", "cuda", "cpu"]: Device name, like 'cuda' or 'cpu'.

    Examples:
        >>> torch.cuda.is_available = lambda: True
        >>> torch.backends.mps.is_available = lambda: False
        >>> get_device_name()
        'cuda'

        >>> torch.cuda.is_available = lambda: False
        >>> torch.backends.mps.is_available = lambda: True
        >>> get_device_name()
        'mps'

        >>> torch.cuda.is_available = lambda: False
        >>> torch.backends.mps.is_available = lambda: False
        >>> get_device_name()
        'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"