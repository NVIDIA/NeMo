---
title: "Fast Entropy-Based Methods of Word-Level Confidence Estimation for End-to-End Automatic Speech Recognition"
author: [Aleksandr Laptev, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2023-01-27

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://ieeexplore.ieee.org/abstract/document/10022960

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Fast Entropy-Based Methods of Word-Level Confidence Estimation for End-to-End Automatic Speech Recognition](https://ieeexplore.ieee.org/abstract/document/10022960)

This paper presents a class of new fast non-trainable entropy-based confidence estimation methods for automatic speech recognition. We show how per-frame entropy values can be normalized and aggregated to obtain a confidence measure per unit and per word for Connectionist Temporal Classification (CTC) and Recurrent Neural Network Transducer (RNN-T) models. Proposed methods have similar computational complexity to the traditional method based on the maximum per-frame probability, but they are more adjustable, have a wider effective threshold range, and better push apart the confidence distributions of correct and incorrect words. We evaluate the proposed confidence measures on LibriSpeech test sets, and show that they are up to 2 and 4 times better than confidence estimation based on the maximum per-frame probability at detecting incorrect words for Conformer-CTC and Conformer-RNN-T models, respectively.

<!-- more -->

