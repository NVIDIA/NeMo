---
title: "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition"
author: [Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh Balam, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2023-09-30

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/2305.05084

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

Conformer-based models have become the dominant end-to-end architecture for speech processing tasks. With the objective of enhancing the conformer architecture for efficient training and inference, we carefully redesigned Conformer with a novel downsampling schema. The proposed model, named Fast Conformer(FC), is 2.8x faster than the original Conformer, supports scaling to Billion parameters without any changes to the core architecture and also achieves state-of-the-art accuracy on Automatic Speech Recognition benchmarks. To enable transcription of long-form speech up to 11 hours, we replaced global attention with limited context attention post-training, while also improving accuracy through fine-tuning with the addition of a global token. Fast Conformer, when combined with a Transformer decoder also outperforms the original Conformer in accuracy and in speed for Speech Translation and Spoken Language Understanding.

<!-- more -->

