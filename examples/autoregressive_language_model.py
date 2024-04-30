"""
Fully convolutional autoregressive language model.
"""
# TODO: also use null terminator
import argparse
import datetime
import gc
import itertools
import logging
import math
import mmap
import os
import re
from contextlib import contextmanager
from functools import partial
from typing import Tuple, Callable, Collection, Any, List, Union

import datasets
import imageio
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
# from adan import Adan  # https://github.com/sail-sg/Adan
from matplotlib.axes import Axes
from thop import profile, clever_format
from thop.vision.basic_hooks import count_convNd, zero_ops, count_normalization, count_adap_avgpool
from thop.vision.calc_func import calculate_conv2d_flops
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import torch
from torch import Tensor
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import tiktoken

import masked_torch.nn as mnn


logger = logging.getLogger(__name__)
# enc = tiktoken.encoding_for_model("gpt-4")
enc = tiktoken.encoding_for_model("gpt-2")


def tokenize(string: str) -> Tensor:
    """
    :return: Tensor of shape (N) of type torch.long, where N depends on the length of the input string.
    """
    # return torch.tensor(list(string.encode('utf-8')))
    # return torch.tensor(list([ord(c) for c in string]))
    return torch.tensor(enc.encode(string))


def detokenize(tensor: Tensor) -> str:
    # return bytes(tensor.tolist()).decode('utf-8', errors='ignore')
    # return ''.join([chr(c) for c in tensor])
    return enc.decode(tensor.tolist())


class AutoregressiveLanguageModel(nn.Module):
    def __init__(
            self,
            depth: int,
            scale: int,
            base: int,
            downscale_factor: int,
            activation: Callable[[], nn.Module] = nn.ReLU,
            sequence_dim: int = 1024,
            num_embeddings: int = 256,
            embedding_dim: int = 128,
            pad_direction: str = "left",
            momentum: float = 0.03,
            eps: float = 1e-5,
    ) -> None:
        assert pad_direction in ("left", "right"), f"{pad_direction=}"
        assert pad_direction == "left", f"Only pad_direction='left' is supported."  # Not fully implemented: this limitation caused by positional encoding, that by the idea should be aligned with the last input item.

        super(AutoregressiveLanguageModel, self).__init__()

        self.multichannel = True
        self.pad_direction = pad_direction

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, _freeze=True)
        pe = torch.randn((embedding_dim, sequence_dim))
        # pe = torch.zeros((embedding_dim, sequence_dim))
        if pad_direction == "right":
            self.pe = nn.Parameter(pe, requires_grad=False)
        elif pad_direction == "left":
            self.pe = nn.Parameter(pe.flip(1), requires_grad=False)

        self.sequential = mnn.MaskedSequential()
        self.sequential.append(mnn.ApplyMask())

        channels_list = [embedding_dim] + [int(scale * base ** i) for i in range(depth)]

        out_channels = 0
        for i, (in_channels, out_channels) in enumerate(zip(channels_list[:-1], channels_list[1:])):
            if i != 0:  # already normalized after embedding
                self.sequential.append(mnn.MaskedBatchNorm1d(in_channels, eps=eps, momentum=momentum))
            self.sequential.append(mnn.MaskedUnshuffle1d(downscale_factor))
            self.sequential.append(mnn.MaskedConv1d(in_channels=in_channels * downscale_factor ** 1, out_channels=out_channels, kernel_size=3, padding=1, bias=True))
            if i < depth - 1:
                self.sequential.append(mnn.Masked(activation()))

        self.sequential.append(mnn.MaskedSpatialMean())

        self.head = nn.Linear(out_channels, num_embeddings, bias=True)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        :param input: Sequential input data of shape (N, L) of type torch.long, where N is the batch size and L is the sequence length.
        :param lengths: Spatial input data size of shape (N,) of type torch.long, where N is the batch size. Represents the length of the sequence in every batch element from the input.
        :return: Logits of next item in sequence of shape (N, C) of type torch.float32, where N is the batch size, and C is the number of classes.
        """
        batch_size, max_sequence_length = input.shape

        x = self.embedding(input)  # (N, L) -> (N, L, C)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)

        pe = self.pe[:, -max_sequence_length:][None].repeat(batch_size, 1, 1)  # (C, L) -> (N, C, L)

        x = x + pe

        if lengths is None:
            mask = None
        else:
            # Create a range tensor that is the same size as the mask's length dimension
            range_tensor = torch.arange(max_sequence_length, device=x.device)[None].repeat(batch_size, 1)

            # Compare range_tensor with lengths to determine where to place the ones in the mask
            if self.pad_direction == "right":
                mask = range_tensor < lengths[:, None]
            elif self.pad_direction == "left":
                mask = range_tensor >= (max_sequence_length - lengths[:, None])
            else:
                raise ValueError(f"{self.pad_direction=}")

            # Finally, make mask for convolutions
            mask = mask[:, None].repeat(1, x.shape[1], 1).to(x.dtype)

        x, mask = self.sequential(x, mask)

        x = self.head(x)

        return x


class MemoryMapSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str, window: int = 1, stride: int = 1):
        self.filepath = filepath
        self.window = window
        self.stride = stride
        self.mm = self.create_mmap()
        self.data_len = self.calculate_data_length()
        # self.data_len = len(self.mm)

    def calculate_data_length(self):
        with open(self.filepath, 'rb') as f:
            f.seek(0, os.SEEK_END)
            length = f.tell()
        return length

    def create_mmap(self):
        file = open(self.filepath, 'r+b')
        mm = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)
        return mm

    def __len__(self):
        return (self.data_len - self.window) // self.stride + 1

    def __getitem__(self, index):
        index = index * self.stride
        self.mm.seek(index)
        x = self.mm.read(self.window)
        return tokenize(x)

    def __del__(self):
        self.mm.close()


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            string: torch.Tensor,
            window_size_min: int,
            window_size_max: int,
    ) -> None:
        assert window_size_min <= window_size_max, f"{window_size_min=}, {window_size_max=}"
        self.string = string
        self.window_size_min = window_size_min
        self.window_size_max = window_size_max
        self.length = max(0, len(self.string) - self.window_size_min)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sequence = self.string[max(0, idx+self.window_size_min-self.window_size_max):idx+self.window_size_min]
        target = self.string[idx+self.window_size_min]
        return {"sequence": sequence, "target": target}


class CollectionSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            strings: Collection[torch.Tensor],
            window_size_min: int,
            window_size_max: int,
    ) -> None:
        assert window_size_min <= window_size_max, f"{window_size_min=}, {window_size_max=}"
        self.strings = strings
        self.window_size_min = window_size_min
        self.window_size_max = window_size_max
        self.cumulative_lengths = self._get_cumulative_lengths()

    def _get_cumulative_lengths(self):
        lengths = [max(0, string.size(0) - self.window_size_min) for string in self.strings]
        cumulative_lengths = [0] + list(itertools.accumulate(lengths))
        return cumulative_lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which string the idx belongs to
        string_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx) - 1

        # Adjust idx to be relative to the start of the found string
        if string_idx > 0:
            idx -= self.cumulative_lengths[string_idx]
        string = self.strings[string_idx]

        # Apply windowing within the found string
        sequence = string[max(0, idx+self.window_size_min-self.window_size_max):idx+self.window_size_min]
        target = string[idx+self.window_size_min]

        return {"sequence": sequence, "target": target}


class MixedSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset]) -> None:
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self):
        cumulative_sizes = [0]
        for dataset in self.datasets:
            cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = next(i for i, total in enumerate(self.cumulative_sizes) if total > idx) - 1
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


def make_pad_collate(dividable_by: int = 1, pad_direction: str = "left"):
    assert pad_direction in ("left", "right"), f"{pad_direction=}"

    def pad_collate(batch):
        # Find the maximum length of all sequences
        max_length = max(example["sequence"].shape[-1] for example in batch)

        # Align the length of all sequences to be dividable by `dividable_by`
        aligned_max_length = max_length + (-max_length % dividable_by)

        for example in batch:
            sequence_length = example["sequence"].shape[-1]
            if pad_direction == "right":
                example["sequence"] = F.pad(example["sequence"], pad=(0, aligned_max_length - sequence_length))
            elif pad_direction == "left":
                example["sequence"] = F.pad(example["sequence"], pad=(aligned_max_length - sequence_length, 0))
            example["sequence_length"] = sequence_length

        # Stack all sequences into a single tensor
        result = {
            "sequence": torch.stack(tuple(example["sequence"] for example in batch)),
            "sequence_lengths": torch.tensor(tuple(example["sequence_length"] for example in batch)),
            "target": torch.stack(tuple(example["target"] for example in batch))
        }
        return result
    return pad_collate


def log_epoch_stats(logger, writer, epoch, train_losses, valid_losses, other):
    """
    Logs the statistics of the current epoch in a formatted grid.

    :param logger: Logger object for logging the information.
    :param epoch: Current epoch number.
    :param train_losses: Dictionary of training loss values.
    :param valid_losses: Dictionary of validation loss values.
    """
    # Define keys
    keys = ['epoch'] + list(other.keys()) + [f'train/{k}' for k in train_losses.keys()] + [f'valid/{k}' for k in valid_losses.keys()]

    # Determine column widths based on key lengths, ensuring a minimum of 10 characters
    column_widths = [max(len(k), 10) for k in keys]

    # Header format
    header_fmt = ', '.join([f'{{:>{w}}}' for w in column_widths])

    # Value format
    values_fmt_list = []
    for i, key in enumerate(keys):
        if key == 'epoch':
            fmt = f'{{:{column_widths[i]}}}'
        else:
            # fmt = f'{{:{column_widths[i]}.6f}}'
            fmt = f'{{:{column_widths[i]}.2e}}'

        values_fmt_list.append(fmt)
    values_fmt = ', '.join(values_fmt_list)

    # Log header
    logger.info(header_fmt.format(*keys))

    # Prepare and log values, handling missing 'train_loss'
    values = [epoch] + list(other.values()) + [train_losses.get(k, '') for k in train_losses] + [valid_losses.get(k, 0.0) for k in valid_losses]
    logger.info(values_fmt.format(*values))
    logger.info('')  # For spacing

    # Summary
    if writer is not None:
        for k, v in train_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in other.items():
            writer.add_scalar(k, v, epoch)
        for k, v in valid_losses.items():
            writer.add_scalar(f"valid/{k}", v, epoch)

        # Save writer
        writer.flush()


def main(args):
    # torch.autograd.set_detect_anomaly(True)

    checkpoints_dir = "checkpoints/alm"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth = 4
    scale = 64
    base = 2
    downscale_factor = 2
    activation = nn.ReLU

    # Model parameters
    pad_direction = "left"
    # vocab_size = 256  # UTF-8
    # vocab_size = 128  # ASCII
    # vocab_size = 2**21  # Unicode code points
    vocab_size = enc.n_vocab

    context_len = 128  # depth = 7
    # context_len = 64  # depth = 6
    # context_len = 32  # depth = 5
    # context_len = 16  # depth = 4
    # context_len = 4
    resume = False
    # resume = True
    load_best = False

    # Training parameters
    max_epochs = 100
    batch_size = 1024
    # batch_size = 2
    # batch_size = 3

    # dividable_by = context_len
    dividable_by = downscale_factor ** depth

    if args.inference:
        max_epochs = 0
        # max_epochs = 1
        resume = True
        load_best = True
        writer = None
    else:
        writer = SummaryWriter()

    ############################################################
    # Load the dataset
    ############################################################
    if not os.path.exists('root/tiny_shakespeare/validation.txt'):
        train_dataset = datasets.load_dataset('tiny_shakespeare', split=datasets.Split.TRAIN)['text'][0].encode('utf-8')
        valid_dataset = datasets.load_dataset('tiny_shakespeare', split=datasets.Split.VALIDATION)['text'][0].encode('utf-8')
        os.makedirs("root/tiny_shakespeare", exist_ok=True)
        with open('root/tiny_shakespeare/train.txt', 'wb') as f:
            f.write(train_dataset)
        with open('root/tiny_shakespeare/validation.txt', 'wb') as f:
            f.write(valid_dataset)

    with open('root/tiny_shakespeare/train.txt', 'r') as f:
        train_dataset = f.read()
    with open('root/tiny_shakespeare/validation.txt', 'r') as f:
        valid_dataset = f.read()

    # train_dataset = "The quick brown fox jumps over the lazy dog. 01234"
    # train_dataset = "! " * batch_size
    # valid_dataset = train_dataset

    train_dataset, valid_dataset = map(tokenize, [train_dataset, valid_dataset])

    train_dataset = SlidingWindowDataset(train_dataset, 1, context_len)
    valid_dataset = SlidingWindowDataset(valid_dataset, 1, context_len)
    # train_dataset = SlidingWindowDataset(train_dataset, context_len, context_len)
    # valid_dataset = SlidingWindowDataset(valid_dataset, context_len, context_len)

    # dataset = datasets.load_dataset("roneneldan/TinyStories")  # https://huggingface.co/datasets/roneneldan/TinyStories

    # train_dataset_stories = [tokenize(story) for story in tqdm(dataset['train']['text'], desc=f"roneneldan/TinyStories/train")]
    # valid_dataset_stories = [tokenize(story) for story in tqdm(dataset['validation']['text'], desc=f"roneneldan/TinyStories/valid")]

    # train_dataset_stories = CollectionSlidingWindowDataset(train_dataset_stories, 1, context_len)
    # valid_dataset_stories = CollectionSlidingWindowDataset(valid_dataset_stories, 1, context_len)

    # valid_dataset = MixedSlidingWindowDataset([valid_dataset, valid_dataset_stories])
    # train_dataset = MixedSlidingWindowDataset([train_dataset, train_dataset_stories, valid_dataset_stories])
    # train_dataset = MixedSlidingWindowDataset([train_dataset, valid_dataset_stories])

    gc.collect()

    # train_dataset = CollectionSlidingWindowDataset(
    #     dataset['train']['text'], 1, context_len, transform, True)
    # valid_dataset = CollectionSlidingWindowDataset(
    #     dataset['validation']['text'], 1, context_len, transform, True)

    # # dataset = datasets.load_dataset("shahules786/orca-chat")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_pad_collate(dividable_by), drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=make_pad_collate(dividable_by))

    model = AutoregressiveLanguageModel(
        depth=depth,
        scale=scale,
        base=base,
        downscale_factor=downscale_factor,
        activation=activation,
        num_embeddings=vocab_size,
        pad_direction=pad_direction,
    )
    model.to(device)
    logger.debug(f"{model=}")

    pbar_epoch = tqdm(total=max_epochs, unit="epoch", position=0, leave=True, desc="Epochs")
    pbar_train = tqdm(total=len(train_loader), unit="batch", position=1, leave=True, desc="Train")
    pbar_valid = tqdm(total=len(valid_loader), unit="batch", position=2, leave=True, desc="Valid")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = Adan(model.parameters(), lr=5e-3, weight_decay=0)

    def criterion(input, output, target):
        ce = F.cross_entropy(output, target)

        losses = {
            "ce": ce,
            "error": (torch.argmax(output, dim=-1) != target).float().mean().item()
        }

        names = ["ce"]
        loss = sum([losses[name] for name in names], start=torch.tensor(0.0))

        return loss, losses

    epoch = 0
    step = 0
    scheduler = None
    best_valid_loss = math.inf

    if resume:
        if load_best:
            data = torch.load(f"{checkpoints_dir}/best.pth")
        else:
            data = torch.load(f"{checkpoints_dir}/last.pth")
        print(f"{load_best}=")

        # model_state_dict = data["model"]
        # # Remove total_ops and total_params from the state dict (i.e. embedding.total_ops, sequential.0.total_params)
        # model_state_dict = {k: v for k, v in model_state_dict.items() if re.search(r"total_(ops|params)", k) is None}

        model.load_state_dict(data["model"])
        # model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(data["optimizer"])
        if data["scheduler"] is not None:
            scheduler.load_state_dict(data["scheduler"])
        best_valid_loss = data["best_valid_loss"]
        step = data['step']
        epoch = data['epoch']
        pbar_epoch.n = epoch

    while epoch < max_epochs:
        pbar_train.reset()
        train_losses = {}

        model.train()

        for data_i, data in enumerate(train_loader):
            # Skip training this epoch to validate the model first
            if epoch == 0:
                break

            sequence, sequence_lengths, target = data["sequence"], data["sequence_lengths"], data["target"]
            sequence, sequence_lengths, target = sequence.to(device), sequence_lengths.to(device), target.to(device)

            output = model(input=sequence, lengths=sequence_lengths)

            loss, losses = criterion(data, output, target)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses = {
                k: train_losses.get(k, 0) + ((v.item() if isinstance(v, Tensor) else v) - train_losses.get(k, 0)) / (data_i + 1)
                for k, v in (losses | {"loss": loss}).items()
            }
            pbar_train.set_postfix(train_losses)
            pbar_train.update(1)

            if writer is not None:
                other = {'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]}
                for k, v in (losses | {"loss": loss, **other}).items():
                    writer.add_scalar("closure/" + k, (v.item() if isinstance(v, Tensor) else v), step)
            step += 1

        pbar_valid.reset()

        model.eval()
        valid_losses = {}

        with torch.no_grad():
            for data_i, data in enumerate(valid_loader):
                sequence, sequence_lengths, target = data["sequence"], data["sequence_lengths"], data["target"]
                sequence, sequence_lengths, target = sequence.to(device), sequence_lengths.to(device), target.to(device)

                output = model(input=sequence, lengths=sequence_lengths)
                loss, losses = criterion(data, output, target)

                valid_losses = {
                    k: valid_losses.get(k, 0) + ((v.item() if isinstance(v, Tensor) else v) - valid_losses.get(k, 0)) / (data_i + 1)
                    for k, v in (losses | {"loss": loss}).items()
                }
                pbar_valid.set_postfix(valid_losses)
                pbar_valid.update(1)

        # Define header and format strings based on keys
        other = {'train/lr': optimizer.param_groups[0]["lr"]}
        log_epoch_stats(logger, writer, epoch, train_losses, valid_losses, other)

        if epoch > 0:
            scheduler is not None and scheduler.step()

        is_best = valid_losses["loss"] < best_valid_loss
        if is_best:
            best_valid_loss = valid_losses["loss"]

        # Save the model
        if not args.inference:
            os.makedirs(f"{checkpoints_dir}/", exist_ok=True)
            data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_valid_loss": best_valid_loss,
                "epoch": epoch,
                "step": step,
            }
            if os.path.exists(f"{checkpoints_dir}/~last.pth"):
                os.unlink(f"{checkpoints_dir}/~last.pth")
            torch.save(data, f"{checkpoints_dir}/~last.pth")
            if os.path.exists(f"{checkpoints_dir}/last.pth"):
                os.unlink(f"{checkpoints_dir}/last.pth")
            os.rename(f"{checkpoints_dir}/~last.pth", f"{checkpoints_dir}/last.pth")
            if is_best:
                if os.path.exists(f"{checkpoints_dir}/best.pth"):
                    os.unlink(f"{checkpoints_dir}/best.pth")
                os.link(f"{checkpoints_dir}/last.pth", f"{checkpoints_dir}/best.pth")
                with open(f"{checkpoints_dir}/best.txt", "w") as f:
                    f.write(f"{datetime.datetime.now()=}, {epoch=}\n")

        pbar_epoch.update(1)
        epoch += 1

    # Parameters and MACs
    def count_masked_convNd(m, x, y):
        assert isinstance(x, tuple) and len(x) == 2 and isinstance(y, tuple) and len(y) == 2
        m.total_ops += calculate_conv2d_flops(
            input_size=list(x[0].shape),
            output_size=list(y[0].shape),
            kernel_size=list(m.weight.shape),
            groups=m.groups,
            bias=m.bias
        )
        m.total_ops += calculate_conv2d_flops(
            input_size=list(x[0].shape),
            output_size=list(y[0].shape),
            kernel_size=[1] * len(m.weight.shape),
            groups=m.groups,
            bias=False
        )

    def count_masked_normalization(m, x, y):
        return count_normalization(m, x[0], y[0])

    custom_ops = {
        # mnn.PartialConv1d: count_masked_convNd,
        # mnn.PartialConv2d: count_masked_convNd,
        # mnn.PartialConv3d: count_masked_convNd,
        mnn.MaskedConv1d: count_masked_convNd,
        mnn.MaskedConv2d: count_masked_convNd,
        mnn.MaskedConv3d: count_masked_convNd,
        mnn.MaskedBatchNorm1d: count_masked_normalization,
        mnn.MaskedBatchNorm2d: count_masked_normalization,
        mnn.MaskedBatchNorm3d: count_masked_normalization,
        # mnn.MaskedMean1d: count_adap_avgpool,
        # mnn.SequenceUnshuffle: zero_ops,
    }
    macs, params = profile(model, inputs=(
        torch.zeros(1, context_len, dtype=torch.long),
        torch.ones(1, dtype=torch.long) * context_len,
    ), custom_ops=custom_ops)
    fmacs, fparams = clever_format([macs, params], "%.3f")
    logger.info(f"Params: {fparams}, MACs: {fmacs}, MACs/Param: {macs / (params + 1e-5):.3f}")

    model.eval()
    with torch.no_grad():
        if args.inference:
            # Sample from model
            predict_len = 320
            temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1]
            min_p = 0.01
            prompts = [
                "The",
                "First Citizen:\nBefore we proceed any further, hear me",
            ]

            for prompt in prompts:
                for temperature in temperatures:
                    # predicted = torch.empty(0, dtype=torch.long)
                    predicted = tokenize(prompt)
                    context = tokenize(prompt)
                    for i in range(predict_len):
                        sequence = context[-context_len:]
                        sequence_len = sequence.shape[0]
                        if pad_direction == "right":
                            sequence = F.pad(sequence, (0, context_len - sequence_len))
                        elif pad_direction == "left":
                            sequence = F.pad(sequence, (context_len - sequence_len, 0))
                        else:
                            raise ValueError(f"{pad_direction=}")
                        sequence_lengths = torch.tensor(sequence_len)

                        output = model(input=sequence[None], lengths=sequence_lengths[None])
                        output = output[0]

                        if temperature <= 0:
                            # Greedy sampling
                            next_char = torch.argmax(output, dim=-1)
                        else:
                            # Stochastic sampling
                            probs = F.softmax(output / temperature, dim=-1)
                            probs = torch.where(probs < min_p, torch.zeros_like(probs), probs)
                            probs /= probs.sum()  # Softmax
                            if not torch.isfinite(probs).any():
                                next_char = torch.argmax(output, dim=-1)
                            else:
                                next_char = torch.multinomial(probs, num_samples=1)
                        # print(f"{context=}")
                        # print(f"{predicted=}")
                        # print(f"{next_char=}")
                        context = torch.tensor(context.tolist() + [next_char.item()])
                        predicted = torch.tensor(predicted.tolist() + [next_char.item()])

                    logger.info(f"temperature: {temperature:.2f}, predicted: {detokenize(predicted)}")

        if args.inference:
            def visualize_reduce_channels(x: Tensor) -> Tensor:
                # return x[0]
                return x.mean(dim=0)
                # return x.std(dim=0, unbiased=False)
                # return ((x - x.mean()) ** 3).mean(dim=0) / (x.std(unbiased=False) ** 3 + eps)
                # return ((x - x.mean()) ** 4).mean(dim=0) / (x.std(unbiased=False) ** 4 + eps)

            def conv_predicate(name: str, module: nn.Module) -> bool:
                if activation is not None:
                    return isinstance(module, nn.ReLU)
                return isinstance(module, nn.Conv1d)

            def mconv_predicate(name: str, module: nn.Module) -> bool:
                if activation is not None:
                    return isinstance(module, mnn.Masked) and isinstance(module.module, nn.ReLU)
                return isinstance(module, nn.Conv1d)

            visualize_activation = True
            visualize_depth = 4
            text = "First Citizen:\nBefore we proceed any further, hear me speak."
            sequence = tokenize(text)
            sequence_len = sequence.shape[0]
            visualize_sequence_len = min(16, sequence_len, context_len)

            batch_i = 0  # constant as we have only one sequence
            visualization = []

            for i in range(0, visualize_sequence_len):
                sequence_start = max(0, sequence.shape[0] - context_len)
                sequence_stop = sequence.shape[0] - visualize_sequence_len + i
                x = sequence[sequence_start:sequence_stop][None]
                target = sequence[sequence_stop]
                print(f"{detokenize(x[0])=}")

                name, (y, activations) = "mconv", forward_with_activations(model, mconv_predicate, x)

                visualization.append((name, sequence_start, sequence_stop, x, y, activations, target))

            ############################################################

            # Initialize pygame
            pygame.init()

            # Set up the display
            WINDOW_WIDTH = 1600
            WINDOW_HEIGHT = 800
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Attention Visualization")

            # Colors
            WHITE = (255, 255, 255)
            BLACK = (0, 0, 0)
            RED = (255, 0, 0)
            BLUE = (0, 0, 255)
            GRAY = (128, 128, 128)
            LIGHT_GRAY = (192, 192, 192)
            # LIGHT_BLUE = (173, 216, 230)
            LIGHT_BLUE = (135, 206, 250)

            # Mono Font
            FONT_SIZE = 40
            # font = pygame.font.Font(None, FONT_SIZE)
            # font = pygame.sysfont.SysFont("monospace", FONT_SIZE)
            font = pygame.sysfont.SysFont("DejaVuSansMono", FONT_SIZE)
            # font = pygame.sysfont.SysFont("jetbrainsmono", FONT_SIZE)
            # font = pygame.sysfont.SysFont("KawkabMono", FONT_SIZE)
            i = 0
            token_padding = 1
            output_file = "attention_animation.gif"
            frames = []

            # Main loop
            running = True
            while running:
                name, sequence_start, sequence_stop, x, y, activations, target = visualization[i]

                screen.fill(WHITE)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Render tokens
                x_offset = 50
                y_offset = 50
                for sequence_i in range(sequence_len):
                    # Draw the token and box
                    s = sequence[sequence_i].item()
                    text = font.render(detokenize(torch.tensor([s])),
                                       True, BLACK if sequence_i < sequence_stop else LIGHT_GRAY)

                    text_width = text.get_width()
                    text_height = text.get_height()

                    if sequence_i in range(sequence_start, sequence_stop):
                        # Calculate contribution of this token to the target token based on its activation
                        contribution = 0

                        for depth_i in range(0, visualize_depth):
                            layer_output = activations[list(activations.keys())[depth_i]]
                            if isinstance(layer_output, torch.Tensor):
                                layer_activation = layer_output[batch_i]
                                layer_activation_mask = None
                            else:
                                layer_activation = layer_output[0][batch_i]
                                layer_activation_mask = layer_output[1][batch_i]

                            layer_activation_mask = None

                            # Warning: calculating activation statistics does not take into account the mask.
                            mean = layer_activation.mean()
                            std = layer_activation.std(unbiased=False)
                            skewness = ((layer_activation - mean) ** 3).mean() / (std ** 3 + eps)
                            kurtosis = ((layer_activation - mean) ** 4).mean() / (std ** 4 + eps)

                            # off = 1
                            off = 2 ** (visualize_depth - depth_i)
                            # c = torch.abs(visualize_reduce_channels(layer_activation[sequence_i // off] * layer_activation_mask[sequence_i // off] if layer_activation_mask is not None else layer_activation[sequence_i // off]) - mean) / std
                            c = visualize_reduce_channels(layer_activation[sequence_i // off] * layer_activation_mask[sequence_i // off] if layer_activation_mask is not None else layer_activation[sequence_i // off]) / (std * 2)
                            # print(f"{depth_i=}, {c=}")
                            contribution += c.item() / off

                        # Draw a blue box around the token based on its contribution to the target token
                        # print(f"{sequence_i=}, {contribution=}")
                        alpha = max(0, min(255, 255 - 255 * contribution))
                        pygame.draw.rect(screen, (alpha, alpha, 255), (x_offset, y_offset, text_width, text_height), 20)

                    if sequence_i == sequence_stop:
                        # Draw a red box around the target token
                        pygame.draw.rect(screen, RED, (x_offset, y_offset, text_width, text_height), 2)

                    screen.blit(text, (x_offset, y_offset))
                    x_offset += text_width + token_padding

                # Draw logits, sorted by logit value
                probs = F.softmax(y[0], dim=-1)
                probs = probs / probs.sum()

                probs_sorted, probs_indices = probs.sort(descending=True)
                # print(f"{probs_sorted=}, {probs_indices=}, {probs_sorted.sum()=}")

                PBAR_WIDTH = 400
                x_offset = WINDOW_WIDTH // 2
                y_offset = 120
                x_spacing = 20
                y_spacing = 20
                LIMIT = 10
                for prob, prob_i in zip(probs_sorted[:LIMIT], probs_indices[:LIMIT]):
                    try:
                        text = font.render(f"{detokenize(torch.tensor([prob_i]))}", True, BLACK)
                    except:
                        text = font.render(f"\\x{prob_i:02x}", True, RED)
                    text_width = text.get_width()
                    text_height = text.get_height()

                    # Draw progress bar for probability and logit value
                    pygame.draw.rect(screen, LIGHT_BLUE, (x_offset + x_spacing // 2, y_offset, int(PBAR_WIDTH * probs[prob_i].item()), text_height), border_radius=10)
                    # pygame.draw.rect(screen, LIGHT_GRAY, (x_offset + x_spacing // 2, y_offset, PBAR_WIDTH, text_height), 2)

                    # Draw percentage
                    prob_text = font.render(f"{int(probs[prob_i].item() * 100)}%", True, BLACK)
                    screen.blit(prob_text, (x_offset + int(PBAR_WIDTH * probs[prob_i].item() + x_spacing), y_offset))

                    # Draw the token
                    screen.blit(text, (x_offset - text_width - x_spacing // 2, y_offset))
                    y_offset += text_height + token_padding + y_spacing

                # Update display
                pygame.display.flip()

                # Move to the next token
                i = (i + 1) % visualize_sequence_len

                if len(frames) < visualize_sequence_len:
                    # Capture the frame
                    frame = pygame.surfarray.array3d(screen)
                    frame = np.flipud(frame)
                    frame = np.rot90(frame, 3)
                    frames.append(frame.copy())

                    if len(frames) == visualize_sequence_len:
                        # Save the animation
                        err = imageio.mimsave(output_file, frames, fps=2/3, loop=0)
                        print(f"{err=}, {output_file=}")
                else:
                    # Add a delay for visualization
                    pygame.time.delay(1500)

            ############################################################

            fig, axs = plt.subplots(nrows=visualize_sequence_len, ncols=visualize_depth, figsize=(12, 8), dpi=120)
            # fig, axs = plt.subplots(nrows=visualize_depth, ncols=visualize_sequence_len, figsize=(12, 8), dpi=120)
            axs = axs.flatten()

            for i, (name, sequence_start, sequence_stop, x, y, activations, target) in enumerate(visualization):
                print(f"{list(activations.keys())=}")

                for depth_i in range(visualize_depth):
                    ax_activation = axs[i * visualize_depth + depth_i]
                    # ax_activation = axs[depth_i * visualize_sequence_len + i]

                    layer_output = activations[list(activations.keys())[depth_i]]
                    if isinstance(layer_output, torch.Tensor):
                        layer_activation = layer_output[batch_i]
                        layer_activation_mask = None
                    else:
                        layer_activation = layer_output[0][batch_i]
                        layer_activation_mask = layer_output[1][batch_i]

                    layer_activation_mask = None

                    # assert layer_activation.dim() == 1, f"{layer_activation.dim()=}"

                    # Warning: calculating activation statistics does not take into account the mask.
                    mean = layer_activation.mean()
                    std = layer_activation.std(unbiased=False)
                    skewness = ((layer_activation - mean) ** 3).mean() / (std ** 3 + eps)
                    kurtosis = ((layer_activation - mean) ** 4).mean() / (std ** 4 + eps)
                    print(f"{name=}, {depth_i=}, {mean=}, {std=}, {skewness=}, {kurtosis=}")

                    visualize = visualize_reduce_channels(layer_activation * layer_activation_mask if layer_activation_mask is not None else layer_activation).unsqueeze(0).repeat(2**(visualize_depth - depth_i), 1)
                    if activation and visualize_activation:
                        # Visualize from batch normalization perspective
                        visualize_deviation = std

                        # Use fixed deviation from conv layers for visualization
                        # visualize_deviation = activations_conv[list(activations.keys())[depth_i-1]][batch_i].std(unbiased=False)

                        ax_activation.imshow(visualize.numpy(), vmin=0, vmax=visualize_deviation * 2)
                        ax_activation.set_title(f"{name}[{depth_i}] activation:")
                    else:
                        # Visualize from batch normalization perspective
                        visualize_mean = mean
                        visualize_deviation = std

                        # if depth_i > 0:
                        #     # visualize_mean = activations_conv[list(activations.keys())[depth_i-1]][batch_i].mean()
                        #     visualize_deviation = activations_conv[list(activations.keys())[depth_i-1]][batch_i].std(unbiased=False)

                        ax_activation.imshow(visualize.numpy(), cmap='coolwarm', vmin=visualize_mean-visualize_deviation, vmax=visualize_mean+visualize_deviation)
                        ax_activation.set_title(f"{name}[{depth_i}] output:")

                    # ax_activation.axis("off")
                    ax_activation.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

                    # if layer_activation_mask is not None:
                    #     visualize = visualize_reduce_channels(layer_activation_mask)[None]
                    #     ax_mask = axs[i * visualize_depth + depth_i]
                    #     # cmap = "viridis"
                    #     cmap = "gray"
                    #     ax_mask.imshow(visualize.numpy(), cmap=cmap, vmin=0, vmax=1)
                    #     if depth_i == 0:
                    #         ax_mask.set_title(f"mask:")
                    #     else:
                    #         ax_mask.set_title(f"{name}[{depth_i}] mask:")
                    #     # ax_mask.axis("off")
                    #     ax_mask.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

            plt.show()



def is_stat_layer(name: str, module: torch.nn.Module):
    return isinstance(module, torch.nn.Conv2d) or name == "shuffles.0"
    # return isinstance(module, torch.nn.BatchNorm2d)


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


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.basicConfig(level=logging.INFO, format="[%(relativeCreated)d] [%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Train a ConvRNN model on a synthetic dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--inference", action="store_true", help="Run inference on the validation set.")
    parser.add_argument("--plot_weights", action="store_true")
    parser.add_argument("--plot_activations", action="store_true")

    args = parser.parse_args()

    with logging_redirect_tqdm():
        main(args)
