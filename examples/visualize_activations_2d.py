from contextlib import contextmanager
from functools import partial
from typing import Tuple, Any, Callable

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

import masked_torch.nn as mnn


@contextmanager
def register_hooks(
        model: torch.nn.Module,
        hook: Callable,
        predicate: Callable[[str, torch.nn.Module], bool],
        **hook_kwargs
):
    handles = []
    try:
        for name, module in model.named_modules():
            if predicate(name, module):
                hook: Callable = partial(hook, name=name, **hook_kwargs)
                handle = module.register_forward_hook(hook)
                handles.append(handle)
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def activations_recorder_hook(
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        name: str,
        *,
        storage: dict[str, Any]
):
    if name in storage:
        if isinstance(storage[name], list):
            storage[name].append(output)
        else:
            storage[name] = [storage[name], output]
    else:
        storage[name] = output


def forward_with_activations(
        model: torch.nn.Module,
        predicate: Callable[[str, torch.nn.Module], bool],
        *model_args,
        **model_kwargs,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    storage = {}
    with register_hooks(model, activations_recorder_hook, predicate, storage=storage):
        output = model(*model_args, **model_kwargs)
    return output, storage


def test_it():
    torch.manual_seed(1337)

    in_channels = 3
    downscale_factor = 2
    scale = 1
    base = 2
    depth = 8
    visualize_depth = 8
    eps = 1e-8
    # activation = True
    activation = False
    visualize_activation = True
    # bias = True
    bias = False
    bias_init_normal = False
    # identity = False
    identity = True

    def visualize_reduce_channels(x: Tensor) -> Tensor:
        # return x[0]
        return x.mean(dim=0)
        # return x.std(dim=0, unbiased=False)
        # return ((x - x.mean()) ** 3).mean(dim=0) / (x.std(unbiased=False) ** 3 + eps)
        # return ((x - x.mean()) ** 4).mean(dim=0) / (x.std(unbiased=False) ** 4 + eps)

    conv = []
    for i in range(depth):
        conv.append(nn.PixelUnshuffle(downscale_factor))
        conv.append(nn.Conv2d(
            in_channels=scale * base ** (i + 1) * downscale_factor ** 2 if i > 0 else in_channels * downscale_factor ** 2,
            out_channels=scale * base ** i * downscale_factor ** 2,
            kernel_size=(3, 3), padding=1, bias=bias)
        )
        bias_init_normal and bias and torch.nn.init.normal_(conv[-1].bias)
        activation and conv.append(nn.ReLU())
    conv = nn.Sequential(*conv)

    # pconv = []
    # for i in range(depth):
    #     pconv.append(mnn.MaskedPixelUnshuffle(downscale_factor))
    #     pconv.append(mnn.PartialConv2d(
    #         in_channels=scale * base ** (i + 1) * downscale_factor ** 2 if i > 0 else in_channels * downscale_factor ** 2,
    #         out_channels=scale * base ** i * downscale_factor ** 2,
    #         kernel_size=(3, 3), padding=1, bias=bias, multichannel=True, eps=eps)
    #     )
    #     bias_init_normal and bias and torch.nn.init.normal_(conv[-1].bias)
    #     activation and pconv.append(mnn.Masked(nn.ReLU()))
    # pconv = mnn.MaskedSequential(*pconv)

    pconv = []
    for i in range(depth):
        pconv.append(mnn.MaskedUnshuffle2d(downscale_factor))
        pconv.append(mnn.PartialConv2d(
            in_channels=scale * base ** (i + 1) * downscale_factor ** 2 if i > 0 else in_channels * downscale_factor ** 2,
            out_channels=scale * base ** i * downscale_factor ** 2,
            kernel_size=(3, 3), padding=1, bias=bias, multichannel=True)
        )
        bias_init_normal and bias and torch.nn.init.normal_(conv[-1].bias)
        activation and pconv.append(mnn.Masked(nn.ReLU()))
    pconv = mnn.MaskedSequential(*pconv)

    mconv = []
    for i in range(depth):
        mconv.append(mnn.MaskedUnshuffle2d(downscale_factor))
        mconv.append(mnn.MaskedConv2d(
            in_channels=scale * base ** (i + 1) * downscale_factor ** 2 if i > 0 else in_channels * downscale_factor ** 2,
            out_channels=scale * base ** i * downscale_factor ** 2,
            kernel_size=(3, 3), padding=1, bias=bias)
        )
        bias_init_normal and bias and torch.nn.init.normal_(conv[-1].bias)
        activation and mconv.append(mnn.Masked(nn.ReLU()))
    mconv = mnn.MaskedSequential(*mconv)

    with (torch.no_grad()):
        print(f"{conv=}")
        print(f"{pconv=}")
        print(f"{mconv=}")

        print(f"{list(conv.state_dict().keys())=}")
        print(f"{list(pconv.state_dict().keys())=}")
        print(f"{list(mconv.state_dict().keys())=}")

        pconv.load_state_dict(conv.state_dict())
        mconv.load_state_dict(conv.state_dict())

        x = torch.randn(1, in_channels, downscale_factor**depth, downscale_factor**depth)
        x_mask = torch.ones_like(x)

        # Cut bottom half
        # x_mask[:, :, 128:256, :] = 0

        # Cut bottom right quarter
        # x_mask[:, :, 128:256, 128:256] = 0

        # Cut bottom right quarter for 2/3 of the channels
        # x_mask[:, :2, 128:256, 128:256] = 0

        # Clockwise cut quarters
        x_mask[:, :1, 0:128, 128:256] = 0
        x_mask[:, :2, 128:256, 128:256] = 0
        x_mask[:, :3, 128:256, 0:128] = 0

        # Grid
        x_mask[:, :, 0:256:2, 0:256:2] = 0
        x_mask[:, :, 0:256:3, 0:256:3] = 0
        # x_mask[:, :, 0:256:4, 0:256:4] = 0

        def conv_predicate(name: str, module: nn.Module) -> bool:
            if activation:
                return isinstance(module, nn.ReLU)
            return isinstance(module, nn.Conv2d)

        def mconv_predicate(name: str, module: nn.Module) -> bool:
            if activation:
                return isinstance(module, mnn.Masked) and isinstance(module.module, nn.ReLU)
            return isinstance(module, nn.Conv2d)

        y_conv, activations_conv = forward_with_activations(conv, conv_predicate, x * x_mask)
        (y_mconv, y_mask_mconv), activations_mconv = forward_with_activations(mconv, mconv_predicate, x * x_mask, x_mask)
        (y_pconv, y_mask_pconv), activations_pconv = forward_with_activations(pconv, mconv_predicate, x * x_mask, x_mask)

        print(f"{list(activations_conv.keys())=}")
        print(f"{list(activations_mconv.keys())=}")
        print(f"{list(activations_pconv.keys())=}")

        fig, axs = plt.subplots(nrows=5, ncols=visualize_depth + 1, figsize=(12, 8), dpi=120)
        axs = axs.flatten()

        row_i = 0
        for name, y, y_mask, activations in [
            ("conv", y_conv, None, activations_conv),
            ("mconv", y_mconv, y_mask_mconv, activations_mconv),
            ("pconv", y_pconv, y_mask_pconv, activations_pconv),
        ]:
            batch_i = 0
            for depth_i in range(visualize_depth + 1):
                ax_activation = axs[row_i * (visualize_depth + 1) + depth_i]

                if depth_i == 0:
                    layer_activation = (x * x_mask)[batch_i]
                    layer_activation_mask = x_mask[batch_i]
                else:
                    layer_output = activations[list(activations.keys())[depth_i - 1]]
                    if isinstance(layer_output, torch.Tensor):
                        layer_activation = layer_output[batch_i]
                        layer_activation_mask = None
                    else:
                        layer_activation = layer_output[0][batch_i]
                        layer_activation_mask = layer_output[1][batch_i]

                assert layer_activation.dim() == 3, f"{layer_activation.dim()=}"

                # Warning: calculating activation statistics does not take into account the mask.
                mean = layer_activation.mean()
                std = layer_activation.std(unbiased=False)
                skewness = ((layer_activation - mean) ** 3).mean() / (std ** 3 + eps)
                kurtosis = ((layer_activation - mean) ** 4).mean() / (std ** 4 + eps)
                print(f"{name=}, {depth_i=}, {mean=}, {std=}, {skewness=}, {kurtosis=}")

                visualize = visualize_reduce_channels(layer_activation * layer_activation_mask if layer_activation_mask is not None else layer_activation)
                if activation and visualize_activation and depth_i > 0:
                    # Visualize from batch normalization perspective
                    visualize_deviation = std

                    # Use fixed deviation from conv layers for visualization
                    # visualize_deviation = activations_conv[list(activations.keys())[depth_i-1]][batch_i].std(unbiased=False)

                    ax_activation.imshow(visualize.numpy(), vmin=0, vmax=visualize_deviation * 2)
                    ax_activation.set_title(f"{name}[{depth_i-1}] activation:")
                else:
                    # Visualize from batch normalization perspective
                    visualize_mean = mean
                    visualize_deviation = std

                    # if depth_i > 0:
                    #     # visualize_mean = activations_conv[list(activations.keys())[depth_i-1]][batch_i].mean()
                    #     visualize_deviation = activations_conv[list(activations.keys())[depth_i-1]][batch_i].std(unbiased=False)

                    ax_activation.imshow(visualize.numpy(), cmap='coolwarm', vmin=visualize_mean-visualize_deviation, vmax=visualize_mean+visualize_deviation)
                    ax_activation.set_title(f"{name}[{depth_i-1}] output:")
                if depth_i == 0:
                    ax_activation.set_title(f"input:")
                # ax_activation.axis("off")
                ax_activation.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

                if layer_activation_mask is not None:
                    visualize = visualize_reduce_channels(layer_activation_mask)
                    ax_mask = axs[(2 + row_i) * (visualize_depth + 1) + depth_i]
                    # cmap = "viridis"
                    cmap = "gray"
                    ax_mask.imshow(visualize.numpy(), cmap=cmap, vmin=0, vmax=1)
                    if depth_i == 0:
                        ax_mask.set_title(f"mask:")
                    else:
                        ax_mask.set_title(f"{name}[{depth_i-1}] mask:")
                    # ax_mask.axis("off")
                    ax_mask.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            row_i += 1

        # plt.suptitle(f"Depth {depth_i}")
        plt.show()


if __name__ == '__main__':
    test_it()
