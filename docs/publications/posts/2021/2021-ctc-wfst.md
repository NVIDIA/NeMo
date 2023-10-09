---
title: "CTC Variations Through New WFST Topologies"
author: [Aleksandr Laptev, Somshubra Majumdar, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-10-06

# Optional: Categories
categories: [Automatic Speech Recognition]
continue_url: https://arxiv.org/abs/2110.03098

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [CTC Variations Through New WFST Topologies](https://arxiv.org/abs/2110.03098)

This paper presents novel Weighted Finite-State Transducer (WFST) topologies to implement Connectionist Temporal Classification (CTC)-like algorithms for automatic speech recognition. Three new CTC variants are proposed: (1) the "compact-CTC", in which direct transitions between units are replaced with <epsilon> back-off transitions; (2) the "minimal-CTC", that only adds <blank> self-loops when used in WFST-composition; and (3) the "selfless-CTC" variants, which disallows self-loop for non-blank units. Compact-CTC allows for 1.5 times smaller WFST decoding graphs and reduces memory consumption by two times when training CTC models with the LF-MMI objective without hurting the recognition accuracy. Minimal-CTC reduces graph size and memory consumption by two and four times for the cost of a small accuracy drop. Using selfless-CTC can improve the accuracy for wide context window models.

<!-- more -->

