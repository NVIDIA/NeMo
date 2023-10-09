---
title: "A Fast and Robust BERT-based Dialogue State Tracker for Schema-Guided Dialogue Dataset"
author: [Vahid Noroozi, Yang Zhang, Evelina Bakhturina, Tomasz Kornuta]
author_gh_user: []
readtime: 30
date: 2020-08-27

# Optional: Categories
categories: [Dialog State Tracking]
continue_url: https://arxiv.org/abs/2008.12335

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [A Fast and Robust BERT-based Dialogue State Tracker for Schema-Guided Dialogue Dataset](https://arxiv.org/abs/2008.12335)

Dialog State Tracking (DST) is one of the most crucial modules for goal-oriented dialogue systems. In this paper, we introduce FastSGT (Fast Schema Guided Tracker), a fast and robust BERT-based model for state tracking in goal-oriented dialogue systems. The proposed model is designed for the Schema-Guided Dialogue (SGD) dataset which contains natural language descriptions for all the entities including user intents, services, and slots. The model incorporates two carry-over procedures for handling the extraction of the values not explicitly mentioned in the current user utterance. It also uses multi-head attention projections in some of the decoders to have a better modelling of the encoder outputs. In the conducted experiments we compared FastSGT to the baseline model for the SGD dataset. Our model keeps the efficiency in terms of computational and memory consumption while improving the accuracy significantly. Additionally, we present ablation studies measuring the impact of different parts of the model on its performance. We also show the effectiveness of data augmentation for improving the accuracy without increasing the amount of computational resources.

<!-- more -->

