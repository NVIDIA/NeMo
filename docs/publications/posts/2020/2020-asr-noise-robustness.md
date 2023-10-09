---
title: "Improving Noise Robustness of an End-to-End Neural Model for Automatic Speech Recognition"
author: [Jagadeesh Balam, Jocelyn Huang, Vitaly Lavrukhin, Slyne Deng, Somshubra Majumdar, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2020-10-23

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/2010.12715

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Improving Noise Robustness of an End-to-End Neural Model for Automatic Speech Recognition](https://arxiv.org/abs/2010.12715)

We present our experiments in training robust to noise an end-to-end automatic speech recognition (ASR) model using intensive data augmentation. We explore the efficacy of fine-tuning a pre-trained model to improve noise robustness, and we find it to be a very efficient way to train for various noisy conditions, especially when the conditions in which the model will be used, are unknown. Starting with a model trained on clean data helps establish baseline performance on clean speech. We carefully fine-tune this model to both maintain the performance on clean speech, and improve the model accuracy in noisy conditions. With this schema, we trained robust to noise English and Mandarin ASR models on large public corpora. All described models and training recipes are open sourced in NeMo, a toolkit for conversational AI.

<!-- more -->

