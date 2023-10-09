---
title: "NeMo Inverse Text Normalization: From Development To Production"
author: [Yang Zhang, Evelina Bakhturina, Kyle Gorman, Boris Ginsburg]
author_gh_user: []
readtime: 30
date: 2021-08-30

# Optional: Categories
categories: [(Inverse) Text Normalization]
continue_url: https://www.isca-speech.org/archive/pdfs/interspeech_2021/zhang21ga_interspeech.pdf

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [NeMo Inverse Text Normalization: From Development To Production](https://www.isca-speech.org/archive/pdfs/interspeech_2021/zhang21ga_interspeech.pdf)

Inverse text normalization (ITN) converts spoken-domain automatic speech recognition (ASR) output into written-domain text to improve the readability of the ASR output. Many stateof-the-art ITN systems use hand-written weighted finite-state transducer (WFST) grammars since this task has extremely low tolerance to unrecoverable errors. We introduce an open-source Python WFST-based library for ITN which enables a seamless path from development to production. We describe the specification of ITN grammar rules for English, but the library can be adapted for other languages. It can also be used for writtento-spoken text normalization. We evaluate the NeMo ITN library using a modified version of the Google Text normalization dataset.

<!-- more -->

