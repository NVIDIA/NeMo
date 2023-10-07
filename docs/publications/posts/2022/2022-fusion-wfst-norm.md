---
title: "Shallow Fusion of Weighted Finite-State Transducer and Language Model for Text Normalization"
author: [Evelina Bakhturina, Yang Zhang, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-03-29

# Optional: Categories
categories: [(Inverse) Text Normalization]
continue_url: https://arxiv.org/abs/2203.15917

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Shallow Fusion of Weighted Finite-State Transducer and Language Model for Text Normalization](https://arxiv.org/abs/2203.15917)

Text normalization (TN) systems in production are largely rule-based using weighted finite-state transducers (WFST). However, WFST-based systems struggle with ambiguous input when the normalized form is context-dependent. On the other hand, neural text normalization systems can take context into account but they suffer from unrecoverable errors and require labeled normalization datasets, which are hard to collect. We propose a new hybrid approach that combines the benefits of rule-based and neural systems. First, a non-deterministic WFST outputs all normalization candidates, and then a neural language model picks the best one -- similar to shallow fusion for automatic speech recognition. While the WFST prevents unrecoverable errors, the language model resolves contextual ambiguity. The approach is easy to extend and we show it is effective. It achieves comparable or better results than existing state-of-the-art TN models.

<!-- more -->

