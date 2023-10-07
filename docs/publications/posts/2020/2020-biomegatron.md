---
title: "BioMegatron: Larger Biomedical Domain Language Model"
author: [Hoo-Chang Shin, Yang Zhang, Evelina Bakhturina, Raul Puri, Mostofa Patwary, Mohammad Shoeybi, Raghav Mani]
author_gh_user: []
readtime: 30
date: 2020-11-01

# Optional: Categories
categories: [Natural Language Processing]
continue_url: https://aclanthology.org/2020.emnlp-main.379/

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [BioMegatron: Larger Biomedical Domain Language Model](https://aclanthology.org/2020.emnlp-main.379/)

There has been an influx of biomedical domain-specific language models, showing language models pre-trained on biomedical text perform better on biomedical domain benchmarks than those trained on general domain text corpora such as Wikipedia and Books. Yet, most works do not study the factors affecting each domain language application deeply. Additionally, the study of model size on domain-specific models has been mostly missing. We empirically study and evaluate several factors that can affect performance on domain language applications, such as the sub-word vocabulary set, model size, pre-training corpus, and domain transfer. We show consistent improvements on benchmarks with our larger BioMegatron model trained on a larger domain corpus, contributing to our understanding of domain language model applications. We demonstrate noticeable improvements over the previous state-of-the-art (SOTA) on standard biomedical NLP benchmarks of question answering, named entity recognition, and relation extraction. Code and checkpoints to reproduce our experiments are available at [github.com/NVIDIA/NeMo].

<!-- more -->

