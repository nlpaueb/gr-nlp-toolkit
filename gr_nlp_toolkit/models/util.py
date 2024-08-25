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

