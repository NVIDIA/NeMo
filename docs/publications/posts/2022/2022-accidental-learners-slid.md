---
title: "Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models"
author: [Travis M. Bartley, Fei Jia, Krishna C. Puvvada, Samuel Kriman, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-11-09

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

# [Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models](https://arxiv.org/abs/2211.05103)

In this paper, we extend previous self-supervised approaches for language identification by experimenting with Conformer based architecture in a multilingual pre-training paradigm. We find that pre-trained speech models optimally encode language discriminatory information in lower layers. Further, we demonstrate that the embeddings obtained from these layers are significantly robust to classify unseen languages and different acoustic environments without additional training. After fine-tuning a pre-trained Conformer model on the VoxLingua107 dataset, we achieve results similar to current state-of-the-art systems for language identification. More, our model accomplishes this with 5x less parameters. We open-source the model through the NVIDIA NeMo toolkit.

<!-- more -->

