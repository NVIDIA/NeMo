---
title: "NVIDIA NeMo Offline Speech Translation Systems for IWSLT 2022"
author: [Oleksii Hrinchuk, Vahid Noroozi, Abhinav Khattar, Anton Peganov, Sandeep Subramanian, Somshubra Majumdar, Oleksii Kuchaiev]
author_gh_user: []
readtime: 30
date: 2022-05-01

# Optional: Categories
categories: [Speech Translation]
continue_url: https://aclanthology.org/2022.iwslt-1.18/

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [NVIDIA NeMo Offline Speech Translation Systems for IWSLT 2022](https://aclanthology.org/2022.iwslt-1.18/)

This paper provides an overview of NVIDIA NeMo’s speech translation systems for the IWSLT 2022 Offline Speech Translation Task. Our cascade system consists of 1) Conformer RNN-T automatic speech recognition model, 2) punctuation-capitalization model based on pre-trained T5 encoder, 3) ensemble of Transformer neural machine translation models fine-tuned on TED talks. Our end-to-end model has less parameters and consists of Conformer encoder and Transformer decoder. It relies on the cascade system by re-using its pre-trained ASR encoder and training on synthetic translations generated with the ensemble of NMT models. Our En->De cascade and end-to-end systems achieve 29.7 and 26.2 BLEU on the 2020 test set correspondingly, both outperforming the previous year’s best of 26 BLEU.

<!-- more -->

