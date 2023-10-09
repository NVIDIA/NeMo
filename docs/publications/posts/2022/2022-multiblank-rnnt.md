---
title: "Multi-blank Transducers for Speech Recognition"
author: [Hainan Xu, Fei Jia, Somshubra Majumdar, Shinji Watanabe, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-11-04

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/2211.03541

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Multi-blank Transducers for Speech Recognition](https://arxiv.org/abs/2211.03541)

This paper proposes a modification to RNN-Transducer (RNN-T) models for automatic speech recognition (ASR). In standard RNN-T, the emission of a blank symbol consumes exactly one input frame; in our proposed method, we introduce additional blank symbols, which consume two or more input frames when emitted. We refer to the added symbols as big blanks, and the method multi-blank RNN-T. For training multi-blank RNN-Ts, we propose a novel logit under-normalization method in order to prioritize emissions of big blanks. With experiments on multiple languages and datasets, we show that multi-blank RNN-T methods could bring relative speedups of over +90%/+139% to model inference for English Librispeech and German Multilingual Librispeech datasets, respectively. The multi-blank RNN-T method also improves ASR accuracy consistently. We will release our implementation of the method in the NeMo (\url{this https URL}) toolkit.

<!-- more -->

