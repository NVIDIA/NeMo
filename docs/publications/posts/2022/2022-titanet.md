---
title: "TitaNet: Neural Model for Speaker Representation with 1D Depth-Wise Separable Convolutions and Global Context"
author: [Nithin Rao Koluguri, Taejin Park, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-04-22

# Optional: Categories
categories: [Speaker Recognition]
continue_url: https://ieeexplore.ieee.org/abstract/document/9746806

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [TitaNet: Neural Model for Speaker Representation with 1D Depth-Wise Separable Convolutions and Global Context](https://ieeexplore.ieee.org/abstract/document/9746806)

In this paper, we propose TitaNet, a novel neural network architecture for extracting speaker representations. We employ 1D depth-wise separable convolutions with Squeeze-and-Excitation (SE) layers with global context followed by channel attention based statistics pooling layer to map variable-length utterances to a fixed-length embedding (t-vector). TitaNet is a scalable architecture and achieves state-of-the-art performance on speaker verification task with an equal error rate (EER) of 0.68% on the VoxCeleb1 trial file and also on speaker diarization tasks with diarization error rate (DER) of 1.73% on AMI-MixHeadset, 1.99% on AMI-Lapel and 1.11% on CH109. Furthermore, we investigate various sizes of TitaNet and present a light TitaNet-S model with only 6M parameters that achieve near state-of-the-art results in diarization tasks.

<!-- more -->

