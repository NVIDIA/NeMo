---
title: "Cross-Language Transfer Learning and Domain Adaptation for End-to-End Automatic Speech Recognition"
author: [Jian Luo, Jianzong Wang, Ning Cheng, Edward Xiao, Jing Xiao, Georg Kucsko, Patrick O’Neill, Jagadeesh Balam, Slyne Deng, Adriana Flores, Boris Ginsburg, Jocelyn Huang, Oleksii Kuchaiev, Vitaly Lavrukhin, Jason Li]
author_gh_user: []
readtime: 30
date: 2021-06-09

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://ieeexplore.ieee.org/document/9428334

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Cross-Language Transfer Learning and Domain Adaptation for End-to-End Automatic Speech Recognition](https://ieeexplore.ieee.org/document/9428334)

In this paper, we demonstrate the efficacy of transfer learning and continuous learning for various automatic speech recognition (ASR) tasks using end-to-end models trained with CTC loss. We start with a large pre-trained English ASR model and show that transfer learning can be effectively and easily performed on: (1) different English accents, (2) different languages (from English to German, Spanish, Russian, or from Mandarin to Cantonese) and (3) application-specific domains. Our extensive set of experiments demonstrate that in all three cases, transfer learning from a good base model has higher accuracy than a model trained from scratch. Our results indicate that, for fine-tuning, larger pre-trained models are better than small pre-trained models, even if the dataset for fine-tuning is small. We also show that transfer learning significantly speeds up convergence, which could result in significant cost savings when training with large datasets.

<!-- more -->

