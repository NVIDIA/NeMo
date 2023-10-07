---
title: "A Unified Transformer-based Framework for Duplex Text Normalization"
author: [Tuan Manh Lai, Yang Zhang, Evelina Bakhturina, Boris Ginsburg, Heng Ji]
author_gh_user: []
readtime: 30
date: 2021-08-23

# Optional: Categories
categories: [(Inverse) Text Normalization]
continue_url: https://arxiv.org/abs/2108.09889

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [A Unified Transformer-based Framework for Duplex Text Normalization](https://arxiv.org/abs/2108.09889)

Text normalization (TN) and inverse text normalization (ITN) are essential preprocessing and postprocessing steps for text-to-speech synthesis and automatic speech recognition, respectively. Many methods have been proposed for either TN or ITN, ranging from weighted finite-state transducers to neural networks. Despite their impressive performance, these methods aim to tackle only one of the two tasks but not both. As a result, in a complete spoken dialog system, two separate models for TN and ITN need to be built. This heterogeneity increases the technical complexity of the system, which in turn increases the cost of maintenance in a production setting. Motivated by this observation, we propose a unified framework for building a single neural duplex system that can simultaneously handle TN and ITN. Combined with a simple but effective data augmentation method, our systems achieve state-of-the-art results on the Google TN dataset for English and Russian. They can also reach over 95% sentence-level accuracy on an internal English TN dataset without any additional fine-tuning. In addition, we also create a cleaned dataset from the Spoken Wikipedia Corpora for German and report the performance of our systems on the dataset. Overall, experimental results demonstrate the proposed duplex text normalization framework is highly effective and applicable to a range of domains and languages

<!-- more -->

