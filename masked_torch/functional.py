from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_batch_norm(input: Tensor, mask: Tensor,
                      running_mean: Optional[Tensor], running_var: Optional[Tensor],
                      weight: Optional[Tensor], bias: Optional[Tensor],
                      training: bool, momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.

    Note: Assuming that mask is already applied to the input.

    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when training=False')

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        # mean = (input * mask).sum(_dims) / num_elements  # (C,)
        mean = input.sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)
        # var = ((input - mean[_slice]) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out
