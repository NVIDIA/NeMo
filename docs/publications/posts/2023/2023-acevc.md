---
title: "ACE-VC: Adaptive and Controllable Voice Conversion using Explicitly Disentangled Self-supervised Speech Representations"
author: [Shehzeen Hussain, Paarth Neekhara, Jocelyn Huang, Jason Li, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2023-06-04

# Optional: Categories
categories: [Text to Speech]
continue_url: https://ieeexplore.ieee.org/abstract/document/10094850

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [ACE-VC: Adaptive and Controllable Voice Conversion using Explicitly Disentangled Self-supervised Speech Representations](https://ieeexplore.ieee.org/abstract/document/10094850)

In this work, we propose a zero-shot voice conversion method using speech representations trained with self-supervised learning. First, we develop a multi-task model to decompose a speech utterance into features such as linguistic content, speaker characteristics, and speaking style. To disentangle content and speaker representations, we propose a training strategy based on Siamese networks that encourages similarity between the content representations of the original and pitch-shifted audio. Next, we develop a synthesis model with pitch and duration predictors that can effectively reconstruct the speech signal from its decomposed representation. Our framework allows controllable and speaker-adaptive synthesis to perform zero-shot any-to-any voice conversion achieving state-of-the-art results on metrics evaluating speaker similarity, intelligibility, and naturalness. Using just 10 seconds of data for a target speaker, our framework can perform voice swapping and achieves a speaker verification EER of 5.5% for seen speakers and 8.4% for unseen speakers.

<!-- more -->
