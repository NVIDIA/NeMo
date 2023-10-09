---
title: "Text Mining Drug/Chemical-Protein Interactions using an Ensemble of BERT and T5 Based Models"
author: [Virginia Adams, Hoo-Chang Shin, Carol Anderson, Bo Liu, Anas Abidin]
author_gh_user: []
readtime: 30
date: 2021-11-21

# Optional: Categories
categories: [Natural Language Processing]
continue_url: https://arxiv.org/abs/2111.15617

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Text Mining Drug/Chemical-Protein Interactions using an Ensemble of BERT and T5 Based Models](https://arxiv.org/abs/2111.15617)

In Track-1 of the BioCreative VII Challenge participants are asked to identify interactions between drugs/chemicals and proteins. In-context named entity annotations for each drug/chemical and protein are provided and one of fourteen different interactions must be automatically predicted. For this relation extraction task, we attempt both a BERT-based sentence classification approach, and a more novel text-to-text approach using a T5 model. We find that larger BERT-based models perform better in general, with our BioMegatron-based model achieving the highest scores across all metrics, achieving 0.74 F1 score. Though our novel T5 text-to-text method did not perform as well as most of our BERT-based models, it outperformed those trained on similar data, showing promising results, achieving 0.65 F1 score. We believe a text-to-text approach to relation extraction has some competitive advantages and there is a lot of room for research advancement.

<!-- more -->

