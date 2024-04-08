---
title: "Investigating end-to-end ASR architectures for long form audio transcription"
author: [Nithin Rao Koluguri, Samuel Kriman, Georgy Zelenfroind, Somshubra Majumdar, Dima Rekesh, Vahid Noroozi, Jagadeesh Balam, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2024-04-14

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://ieeexplore.ieee.org/abstract/document/10448309

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [A Chat about Boring Problems: Studying GPT-Based Text Normalization](https://ieeexplore.ieee.org/abstract/document/10447169)

This paper presents an overview and evaluation of some of the end-to-end ASR models on long-form audio. We study three categories of Automatic Speech Recognition(ASR) models based on their core architecture: (1) convolutional, (2) convolutional with squeeze-and-excitation, and (3) convolutional models with attention. We selected one ASR model from each category and evaluated the Word Error Rate, maximum audio length and real-time factor for each model on a variety of long audio benchmarks: Earnings-21 and 22, CORAAL, and TED-LIUM3. The model from the category of self-attention with local attention and global token has the best accuracy compared to other architectures. We also compared models with CTC and RNNT decoders and showed that CTC-based models are more robust and efficient than RNNT on long form audio.

<!-- more -->

