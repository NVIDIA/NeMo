---
title: "TalkNet: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis"
author: [Stanislav Beliaev, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-02-01

# Optional: Categories
categories: [Text to Speech]
continue_url: https://www.isca-speech.org/archive/interspeech_2021/beliaev21_interspeech.html

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [TalkNet: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis](https://www.isca-speech.org/archive/interspeech_2021/beliaev21_interspeech.html)

We propose TalkNet, a non-autoregressive convolutional neural model for speech synthesis with explicit pitch and duration prediction. The model consists of three feed-forward convolutional networks. The first network predicts grapheme durations. An input text is then expanded by repeating each symbol according to the predicted duration. The second network predicts pitch value for every mel frame. The third network generates a mel-spectrogram from the expanded text conditioned on predicted pitch. All networks are based on 1D depth-wise separable convolutional architecture. The explicit duration prediction eliminates word skipping and repeating. The quality of the generated speech nearly matches the best auto-regressive models — TalkNet trained on the LJSpeech dataset got a MOS of 4.08. The model has only 13.2M parameters, almost 2× less than the present state-of-the-art text-to-speech models. The non-autoregressive architecture allows for fast training and inference. The small model size and fast inference make TalkNet an attractive candidate for embedded speech synthesis.

<!-- more -->

