---
title: "Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers"
author: [Cheng-Ping Hsieh, Subhankar Ghosh, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2022-11-01

# Optional: Categories
categories: [Text to Speech]
continue_url: https://arxiv.org/abs/2211.00585

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

Fine-tuning is a popular method for adapting text-to-speech (TTS) models to new speakers. However this approach has some challenges. Usually fine-tuning requires several hours of high quality speech per speaker. There is also that fine-tuning will negatively affect the quality of speech synthesis for previously learnt speakers. In this paper we propose an alternative approach for TTS adaptation based on using parameter-efficient adapter modules. In the proposed approach, a few small adapter modules are added to the original network. The original weights are frozen, and only the adapters are fine-tuned on speech for new speaker. The parameter-efficient fine-tuning approach will produce a new model with high level of parameter sharing with original model. Our experiments on LibriTTS, HiFi-TTS and VCTK datasets validate the effectiveness of adapter-based method through objective and subjective metrics.

<!-- more -->

