---
title: "QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions"
author: [Samuel Kriman, Stanislav Beliaev, Boris Ginsburg, Jocelyn Huang, Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, Yang Zhang]
author_gh_user: []
readtime: 30
date: 2019-10-22

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/1910.10261

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions](https://arxiv.org/abs/1910.10261)

We propose a new end-to-end neural acoustic model for automatic speech recognition. The model is composed of multiple blocks with residual connections between them. Each block consists of one or more modules with 1D time-channel separable convolutional layers, batch normalization, and ReLU layers. It is trained with CTC loss. The proposed network achieves near state-of-the-art accuracy on LibriSpeech and Wall Street Journal, while having fewer parameters than all competing models. We also demonstrate that this model can be effectively fine-tuned on new datasets.

<!-- more -->

