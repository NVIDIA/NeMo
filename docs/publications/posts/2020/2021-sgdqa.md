---
title: "SGD-QA: Fast Schema-Guided Dialogue State Tracking for Unseen Services"
author: [Yang Zhang, Vahid Noroozi, Evelina Bakhturina, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-05-17

# Optional: Categories
categories: [Dialog State Tracking]
continue_url: https://arxiv.org/abs/2105.08049

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [SGD-QA: Fast Schema-Guided Dialogue State Tracking for Unseen Services](https://arxiv.org/abs/2105.08049)

Dialogue state tracking is an essential part of goal-oriented dialogue systems, while most of these state tracking models often fail to handle unseen services. In this paper, we propose SGD-QA, a simple and extensible model for schema-guided dialogue state tracking based on a question answering approach. The proposed multi-pass model shares a single encoder between the domain information and dialogue utterance. The domain's description represents the query and the dialogue utterance serves as the context. The model improves performance on unseen services by at least 1.6x compared to single-pass baseline models on the SGD dataset. SGD-QA shows competitive performance compared to state-of-the-art multi-pass models while being significantly more efficient in terms of memory consumption and training performance. We provide a thorough discussion on the model with ablation study and error analysis.

<!-- more -->

