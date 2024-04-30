#### [Home](https://github.com/ivanstepanovftw/masked_torch) | [Introduction](#masked-torch) | [Why MaskedConv2d?](#why-maskedconv2d) | [Key Differences](#key-differences) | [Visual Comparison](#visual-comparison) | [Current Limitations](#current-limitations) | [Getting Started](#getting-started) | [License](#license) | [How to Cite](#how-to-cite)

# Masked Torch

Welcome to Masked Torch, the home of a PyTorch implementation for the `MaskedConv2d` layer, designed to address the issues presented in typical convolution operations with varying image sizes and hardware acceleration constraints. This project is inspired by the innovative ideas from the paper "Partial Convolution based Padding" by Guilin Liu et al.

## Introduction
In the realm of convolution operations, the main advantage of handling images of different sizes is often compromised due to the limitations imposed by GPU accelerations which necessitate larger batch sizes. Several solutions have emerged, though none without their drawbacks. The `MaskedConv2d` layer provided in this repository is refined to support better statistical handling and normalization of multi-channel masks without altering the image distribution as seen in conventional methods.

## Why MaskedConv2d?
`PartialConv2d` layers, as discussed in various implementations, tend to disrupt the distribution of output values which can be detrimental in certain use cases. This repository offers `MaskedConv2d` as a drop-in replacement to rectify these issues, promoting stability and efficiency in model training.

## Key Differences
- **Output Distribution:** Unlike `PartialConv2d`, `MaskedConv2d` ensures the output values remain normally distributed.
- **Mask Handling:** `MaskedConv2d` utilizes 1x1 convolution weights for the mask, allowing for direct channel-wise scaling without the blurring effect seen in `PartialConv2d`.
- **Efficiency:** `MaskedConv2d` is optimized for no-mask scenarios which accelerates training without compromising on performance.

## Visual Comparison
A comparative visualization of activation statistics (per-channel mean) between conventional convolution (conv), PartialConv (pconv), and MaskedConv (mconv). It highlights how `MaskedConv2d` maintains a more consistent and stable distribution across the layers:

![Visual Comparison](https://i.imgur.com/8ifNtrP.png)

_Reproduce this visualization by running [visualize_activations_2d.py](examples/visualize_activations_2d.py)._

## Current Limitations
- Lazy module implementation is pending.
- Exporting to ONNX format might encounter issues due to the use of inplace operators.

## Getting Started
To get started, clone this repository and integrate it into your PyTorch projects. Play with examples, such as:
- autoregressive language model [autoregressive_language_model.py](examples/autoregressive_language_model.py)
  
  ![attention visualization](https://i.imgur.com/w6wgvoA.gif)
  
  _Attention visualization in autoregressive language model._

[//]: # (- object detection [object_detection.py]&#40;examples/object_detection.py&#41;)

to see the benefits of using masked convolution.

## License
This project is licensed under the Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0).

## How to Cite
If you find this implementation useful in your research, please consider citing it:
```bibtex
@misc{mconv2024,
    title = {Masked Convolution for PyTorch},
    author = {Ivan Stepanov},
    year = {2024},
    howpublished = {\url{https://github.com/ivanstepanovftw/masked_torch}},
    note = {Accessed: April 30, 2024}
}
```
