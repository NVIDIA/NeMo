---
title: "Damage Control During Domain Adaptation for Transducer Based Automatic Speech Recognition"
author: [Somshubra Majumdar, Shantanu Acharya, Vitaly Lavrukhin, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2023-01-09

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://ieeexplore.ieee.org/abstract/document/10023219

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Damage Control During Domain Adaptation for Transducer Based Automatic Speech Recognition](https://ieeexplore.ieee.org/abstract/document/10023219)

Automatic speech recognition models are often adapted to improve their accuracy in a new domain. A potential drawback of model adaptation to new domains is catastrophic forgetting, where the Word Error Rate on the original domain is significantly degraded. This paper addresses the situation when we want to simultaneously adapt automatic speech recognition models to a new domain and limit the degradation of accuracy on the original domain without access to the original training dataset. We propose several techniques such as a limited training strategy and regularized adapter modules for the Transducer encoder, prediction, and joiner network. We apply these methods to the Google Speech Commands and to the UK and Ireland English Dialect speech data set and obtain strong results on the new target domain while limiting the degradation on the original domain.

<!-- more -->

