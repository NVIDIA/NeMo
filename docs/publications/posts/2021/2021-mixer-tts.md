---
title: "Mixer-TTS: non-autoregressive, fast and compact text-to-speech model conditioned on language model embeddings"
author: [Oktai Tatanov, Stanislav Beliaev, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-10-07

# Optional: Categories
categories: [Text to Speech]
continue_url: https://arxiv.org/abs/2110.03584

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [Mixer-TTS: non-autoregressive, fast and compact text-to-speech model conditioned on language model embeddings](https://arxiv.org/abs/2110.03584)

This paper describes Mixer-TTS, a non-autoregressive model for mel-spectrogram generation. The model is based on the MLP-Mixer architecture adapted for speech synthesis. The basic Mixer-TTS contains pitch and duration predictors, with the latter being trained with an unsupervised TTS alignment framework. Alongside the basic model, we propose the extended version which additionally uses token embeddings from a pre-trained language model. Basic Mixer-TTS and its extended version achieve a mean opinion score (MOS) of 4.05 and 4.11, respectively, compared to a MOS of 4.27 of original LJSpeech samples. Both versions have a small number of parameters and enable much faster speech synthesis compared to the models with similar quality.

<!-- more -->

