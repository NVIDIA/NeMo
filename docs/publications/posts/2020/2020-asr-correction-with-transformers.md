---
title: "Correction of Automatic Speech Recognition with Transformer Sequence-To-Sequence Model"
author: [Oleksii Hrinchuk, Mariya Popova, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2020-04-20

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://ieeexplore.ieee.org/abstract/document/9053051

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Correction of Automatic Speech Recognition with Transformer Sequence-To-Sequence Model](https://ieeexplore.ieee.org/abstract/document/9053051)

In this work, we introduce a simple yet efficient post-processing model for automatic speech recognition. Our model has Transformer-based encoder-decoder architecture which "translates" acoustic model output into grammatically and semantically correct text. We investigate different strategies for regularizing and optimizing the model and show that extensive data augmentation and the initialization with pretrained weights are required to achieve good performance. On the LibriSpeech benchmark, our method demonstrates significant improvement in word error rate over the baseline acoustic model with greedy decoding, especially on much noisier dev-other and test-other portions of the evaluation dataset. Our model also outperforms baseline with 6-gram language model re-scoring and approaches the performance of re-scoring with Transformer-XL neural language model.

<!-- more -->

