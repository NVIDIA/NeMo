---
title: "Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation"
author: [Virginia Adams, Sandeep Subramanian, Mike Chrzanowski, Oleksii Hrinchuk, Oleksii Kuchaiev]
author_gh_user: []
readtime: 30
date: 2022-06-02

# Optional: Categories
categories: [Neural Machine Translation]
continue_url: https://arxiv.org/abs/2206.01137

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation](https://arxiv.org/abs/2206.01137)

General translation models often still struggle to generate accurate translations in specialized domains. To guide machine translation practitioners and characterize the effectiveness of domain adaptation methods under different data availability scenarios, we conduct an in-depth empirical exploration of monolingual and parallel data approaches to domain adaptation of pre-trained, third-party, NMT models in settings where architecture change is impractical. We compare data centric adaptation methods in isolation and combination. We study method effectiveness in very low resource (8k parallel examples) and moderately low resource (46k parallel examples) conditions and propose an ensemble approach to alleviate reductions in original domain translation quality. Our work includes three domains: consumer electronic, clinical, and biomedical and spans four language pairs - Zh-En, Ja-En, Es-En, and Ru-En. We also make concrete recommendations for achieving high in-domain performance and release our consumer electronic and medical domain datasets for all languages and make our code publicly available.

<!-- more -->

