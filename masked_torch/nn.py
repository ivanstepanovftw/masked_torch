from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from masked_torch import functional as MF


class ApplyMask(nn.Module):
    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        if mask is not None:
            input = input * mask
        return input, mask


class UnMask(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        if mask is not None:
            input = input / (mask + self.eps)
        return input, mask


class MaskedSequential(nn.Sequential):
    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        for i, module in enumerate(self):
            assert mask.shape == input.shape, f"{mask.shape=}, {input.shape=}"
            input, mask = module(input, mask)
            assert mask.shape == input.shape, f"{mask.shape=}, {input.shape=}"
        return input, mask


class SpatialMean(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        dim = tuple(range(2, input.dim()))
        input = input.mean(dim=dim)
        return input


class MaskedSpatialMean(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        dim = tuple(range(2, input.dim()))
        if mask is not None:
            assert mask.shape == input.shape, f"{mask.shape=}, {input.shape=}"
            input = input.sum(dim=dim) / mask.sum(dim=dim).add_(self.eps)
            mask = mask.mean(dim=dim)
        else:
            input = input.mean(dim=dim)
        return input, mask


class Unshuffle1d(nn.Module):
    def __init__(self, downscale_factor: int):
        super(Unshuffle1d, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        batch_size, channels, length = input.size()
        factor = self.downscale_factor

        # Ensure spatial dimensions are divisible by the downscale factor
        if length % factor:
            raise ValueError("Length must be divisible by the downscale factor")

        # Reshape the input sequence to split it into chunks
        input = input.view(batch_size, channels, length // factor, factor)

        # Transpose and reshape to unshuffle the sequence
        input = input.permute(0, 1, 3, 2).contiguous().view(batch_size, channels * factor, -1)

        return input

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"


class Unshuffle2d(nn.PixelUnshuffle):
    pass


class Unshuffle3d(nn.Module):
    def __init__(self, downscale_factor: int):
        super(Unshuffle3d, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        batch_size, channels, depth, height, width = input.size()
        factor = self.downscale_factor

        # Ensure dimensions are divisible by the downscale factor
        if depth % factor or height % factor or width % factor:
            raise ValueError("Depth, height, and width must be divisible by the downscale factor.")

        # Reshape the input sequence to split it into chunks
        input = input.view(batch_size, channels, depth // factor, factor, height // factor, factor, width // factor, factor)

        # Transpose and reshape to unshuffle the spatial dimensions
        input = input.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous().view(batch_size, channels * factor ** 3, depth // factor, height // factor, width // factor)

        return input

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"


class MaskedUnshuffle1d(Unshuffle1d):
    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        if mask is not None:
            mask = super().forward(mask)
        return super().forward(input), mask


class MaskedUnshuffle2d(Unshuffle2d):
    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        if mask is not None:
            mask = super().forward(mask)
        return super().forward(input), mask


class MaskedUnshuffle3d(Unshuffle3d):
    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        if mask is not None:
            mask = super().forward(mask)
        return super().forward(input), mask


class Masked(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        assert not isinstance(module, (ApplyMask, UnMask, MaskedSequential, SpatialMean, MaskedSpatialMean, Unshuffle1d, Unshuffle2d, Unshuffle3d, MaskedUnshuffle1d, MaskedUnshuffle2d, MaskedUnshuffle3d, Masked, PartialConv1d, PartialConv2d, PartialConv3d, MaskedConv1d, MaskedConv2d, MaskedConv3d, MaskedBatchNorm1d, MaskedBatchNorm2d, MaskedBatchNorm3d)), f"double masking"
        super().__init__()
        self.module = module

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        return self.module(input), mask

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module})"


def _pconv_make_mask_weight(
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        groups: int,
        multichannel: bool,
        device=None,
        dtype=None
) -> Tensor:
    if multichannel:
        mask_weight_shape = (out_channels, in_channels // groups, *kernel_size)
    else:
        mask_weight_shape = (1, 1, *kernel_size)
    mask_weight = torch.ones(mask_weight_shape, device=device, dtype=dtype)
    return mask_weight


def _pconv_validate_mask(
        input: Tensor,
        mask: Tensor | None,
        multichannel: bool
) -> Tensor:
    if mask is None:
        if multichannel:
            mask = torch.ones_like(input)
        else:
            mask = torch.ones(1, 1, *input.shape[2:], device=input.device, dtype=input.dtype)
    else:
        if multichannel:
            mask = mask.expand_as(input)
        else:
            mask = mask.expand(1, 1, *input.shape[2:])
    return mask


class PartialConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            multichannel: bool = False,
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.multichannel = multichannel
        self.eps = eps
        mask_weight = _pconv_make_mask_weight(in_channels, out_channels, self.kernel_size, groups, multichannel, device, dtype)
        self.register_buffer('mask_weight', mask_weight, persistent=False)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor):
        # Convolve as usual without bias
        output = F.conv1d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        with torch.no_grad():
            # Get mask with correct shape for input
            mask = _pconv_validate_mask(input, mask, self.multichannel)

            # Convolve mask on mask_weight
            mask = F.conv1d(mask, self.mask_weight, None, self.stride, self.padding, self.dilation, self.groups if self.multichannel else 1)

            # Calculate mask_ratio for re-weighting and clamp the mask
            mask_kernel_numel = self.mask_weight.data.shape[1:].numel()
            mask_ratio = mask_kernel_numel / (mask + self.eps)
            mask.clamp_max_(1)

        # Apply re-weighting and bias
        output *= mask_ratio
        if self.bias is not None:
            output += self.bias.view(-1, 1)

        # Apply mask
        output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, multichannel={self.multichannel}, eps={self.eps}"


class PartialConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            multichannel: bool = False,
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.multichannel = multichannel
        self.eps = eps
        mask_weight = _pconv_make_mask_weight(in_channels, out_channels, self.kernel_size, groups, multichannel, device, dtype)
        self.register_buffer('mask_weight', mask_weight, persistent=False)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor):
        # Convolve as usual without bias
        output = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        with torch.no_grad():
            # Get mask with correct shape for input
            mask = _pconv_validate_mask(input, mask, self.multichannel)

            # Convolve mask on mask_weight
            mask = F.conv2d(mask, self.mask_weight, None, self.stride, self.padding, self.dilation, self.groups if self.multichannel else 1)

            # Calculate mask_ratio for re-weighting and clamp the mask
            mask_kernel_numel = self.mask_weight.data.shape[1:].numel()
            mask_ratio = mask_kernel_numel / (mask + self.eps)
            mask.clamp_max_(1)

        # Apply re-weighting and bias
        output *= mask_ratio
        if self.bias is not None:
            output += self.bias.view(-1, 1, 1)

        # Apply mask
        output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, multichannel={self.multichannel}, eps={self.eps}"


class PartialConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            multichannel: bool = False,
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.multichannel = multichannel
        self.eps = eps
        mask_weight = _pconv_make_mask_weight(in_channels, out_channels, self.kernel_size, groups, multichannel, device, dtype)
        self.register_buffer('mask_weight', mask_weight, persistent=False)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor):
        # Convolve as usual without bias
        output = F.conv3d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        with torch.no_grad():
            # Get mask with correct shape for input
            mask = _pconv_validate_mask(input, mask, self.multichannel)

            # Convolve mask on mask_weight
            mask = F.conv3d(mask, self.mask_weight, None, self.stride, self.padding, self.dilation, self.groups if self.multichannel else 1)

            # Calculate mask_ratio for re-weighting and clamp the mask
            mask_kernel_numel = self.mask_weight.data.shape[1:].numel()
            mask_ratio = mask_kernel_numel / (mask + self.eps)
            mask.clamp_max_(1)

        # Apply re-weighting and bias
        output *= mask_ratio
        if self.bias is not None:
            output += self.bias.view(-1, 1, 1, 1)

        # Apply mask
        output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, multichannel={self.multichannel}, eps={self.eps}"


class MaskedConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        assert not isinstance(padding, str), f"Untested {padding=}"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        for k in self.kernel_size:
            if k % 2 != 1:
                raise ValueError("Only odd kernel sizes are supported")
        self.eps = eps
        self.mask_padding = (self.padding[0] - self.kernel_size[0] // 2, )

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        # Convolve as usual
        output = F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if mask is not None:
            with torch.no_grad():
                # Create mask_weight of shape (fan_out, fan_in, 1)
                mask_weight = self.weight.abs().sum(dim=2, keepdim=True)

                # Convolve mask
                mask = F.conv1d(mask, mask_weight, None, self.stride, self.mask_padding, self.dilation, self.groups)

                # Normalize the mask by the mask_weight_sum of shape (fan_out, 1, 1), viewed to (1, fan_out, 1)
                mask_weight_sum = mask_weight.sum(dim=1, keepdim=True).view(1, -1, 1)
                mask /= mask_weight_sum.add_(self.eps)

            # Apply mask
            output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eps={self.eps}"


class MaskedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        assert not isinstance(padding, str), f"Untested {padding=}"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        for k in self.kernel_size:
            if k % 2 != 1:
                raise ValueError("Only odd kernel sizes are supported")
        self.eps = eps
        self.mask_padding = (self.padding[0] - self.kernel_size[0] // 2,
                             self.padding[1] - self.kernel_size[1] // 2)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        # Convolve as usual
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if mask is not None:
            with torch.no_grad():
                # Create mask_weight of shape (fan_out, fan_in, 1, 1)
                mask_weight = self.weight.abs().sum(dim=(2, 3), keepdim=True)

                # Convolve mask
                mask = F.conv2d(mask, mask_weight, None, self.stride, self.mask_padding, self.dilation, self.groups)

                # Normalize the mask by the mask_weight_sum of shape (fan_out, 1, 1, 1), viewed to (1, fan_out, 1, 1)
                mask_weight_sum = mask_weight.sum(dim=1, keepdim=True).view(1, -1, 1, 1)
                mask /= mask_weight_sum.add_(self.eps)

            # Apply mask
            output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eps={self.eps}"


class MaskedConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            eps=1e-8,
            device=None,
            dtype=None
    ) -> None:
        assert padding_mode == "zeros", "Only padding_mode='zeros' is supported"
        assert not isinstance(padding, str), f"Untested {padding=!r}"
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        for k in self.kernel_size:
            if k % 2 != 1:
                raise ValueError("Only odd kernel sizes are supported")
        self.eps = eps
        self.mask_padding = (self.padding[0] - self.kernel_size[0] // 2,
                             self.padding[1] - self.kernel_size[1] // 2,
                             self.padding[2] - self.kernel_size[2] // 2)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        # Convolve as usual
        output = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if mask is not None:
            with torch.no_grad():
                # Create mask_weight of shape (fan_out, fan_in, 1, 1, 1)
                mask_weight = self.weight.abs().sum(dim=(2, 3, 4), keepdim=True)

                # Convolve mask
                mask = F.conv3d(mask, mask_weight, None, self.stride, self.mask_padding, self.dilation, self.groups)

                # Normalize the mask by the mask_weight_sum of shape (fan_out, 1, 1, 1, 1), viewed to (1, fan_out, 1, 1, 1)
                mask_weight_sum = mask_weight.sum(dim=1, keepdim=True).view(1, -1, 1, 1, 1)
                mask /= mask_weight_sum.add_(self.eps)

            # Apply mask
            output = output * mask

        return output, mask

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eps={self.eps}"


class _MaskedBatchNorm(_BatchNorm):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: Tensor, mask: Tensor | None = None) -> (Tensor, Tensor | None):
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            ), mask
        else:
            return MF.masked_batch_norm(
                input, mask,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            ), mask


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class MaskedBatchNorm2d(torch.nn.BatchNorm2d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm2d` for details.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class MaskedBatchNorm3d(torch.nn.BatchNorm3d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 5D input
    (a mini-batch of 3D inputs with additional channel dimension).

    See documentation of :class:`~torch.nn.BatchNorm3d` for details.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Mask: :math:`(N, 1, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
