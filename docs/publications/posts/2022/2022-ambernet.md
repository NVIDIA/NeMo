---
title: "A Compact End-to-End Model with Local and Global Context for Spoken Language Identification"
author: [Fei Jia, Nithin Rao Koluguri, Jagadeesh Balam, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-10-27

# Optional: Categories
categories: [Speech Classification]
continue_url: https://arxiv.org/abs/2210.15781

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [A Compact End-to-End Model with Local and Global Context for Spoken Language Identification](https://arxiv.org/abs/2210.15781)

We introduce TitaNet-LID, a compact end-to-end neural network for Spoken Language Identification (LID) that is based on the ContextNet architecture. TitaNet-LID employs 1D depth-wise separable convolutions and Squeeze-and-Excitation layers to effectively capture local and global context within an utterance. Despite its small size, TitaNet-LID achieves performance similar to state-of-the-art models on the VoxLingua107 dataset while being 10 times smaller. Furthermore, it can be easily adapted to new acoustic conditions and unseen languages through simple fine-tuning, achieving a state-of-the-art accuracy of 88.2% on the FLEURS benchmark. Our model is scalable and can achieve a better trade-off between accuracy and speed. TitaNet-LID performs well even on short utterances less than 5s in length, indicating its robustness to input length.

<!-- more -->

