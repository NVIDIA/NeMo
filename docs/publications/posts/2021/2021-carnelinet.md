---
title: "CarneliNet: Neural Mixture Model for Automatic Speech Recognition"
author: [Aleksei Kalinov, Somshubra Majumdar, Jagadeesh Balam, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-07-22

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/2107.10708

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [CarneliNet: Neural Mixture Model for Automatic Speech Recognition](https://arxiv.org/abs/2107.10708)

End-to-end automatic speech recognition systems have achieved great accuracy by using deeper and deeper models. However, the increased depth comes with a larger receptive field that can negatively impact model performance in streaming scenarios. We propose an alternative approach that we call Neural Mixture Model. The basic idea is to introduce a parallel mixture of shallow networks instead of a very deep network. To validate this idea we design CarneliNet -- a CTC-based neural network composed of three mega-blocks. Each mega-block consists of multiple parallel shallow sub-networks based on 1D depthwise-separable convolutions. We evaluate the model on LibriSpeech, MLS and AISHELL-2 datasets and achieved close to state-of-the-art results for CTC-based models. Finally, we demonstrate that one can dynamically reconfigure the number of parallel sub-networks to accommodate the computational requirements without retraining.

<!-- more -->

