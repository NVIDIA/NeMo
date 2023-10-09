---
title: "Leveraging Synthetic Targets for Machine Translation"
author: [Sarthak Mittal, Oleksii Hrinchuk, Oleksii Kuchaiev]
author_gh_user: []
readtime: 30
date: 2023-05-07

# Optional: Categories
categories: [Neural Machine Translation]
continue_url: https://arxiv.org/abs/2305.06155

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Leveraging Synthetic Targets for Machine Translation](https://arxiv.org/abs/2305.06155)

In this work, we provide a recipe for training machine translation models in a limited resource setting by leveraging synthetic target data generated using a large pre-trained model. We show that consistently across different benchmarks in bilingual, multilingual, and speech translation setups, training models on synthetic targets outperforms training on the actual ground-truth data. This performance gap grows bigger with increasing limits on the amount of available resources in the form of the size of the dataset and the number of parameters in the model. We also provide preliminary analysis into whether this boost in performance is linked to ease of optimization or more deterministic nature of the predictions, and whether this paradigm leads to better out-of-distribution performance across different testing domains.

<!-- more -->

