---
title: "SelfVC: Voice Conversion With Iterative Refinement using Self Transformations"
author: [Paarth Neekhara, Shehzeen Hussain, Rafael Valle, Boris Ginsburg, Rishabh Ranjan, Shlomo Dubnov, Farinaz Koushanfar, Julian McAuley]
author_gh_user: []
readtime: 30
date: 2024-05-03

# Optional: Categories
categories: [Voice Conversion, Speech Generation]
continue_url: https://arxiv.org/abs/2310.09653

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [SelfVC: Voice Conversion With Iterative Refinement using Self Transformations](https://arxiv.org/abs/2310.09653)

We propose SelfVC, a training strategy to iteratively improve a voice conversion model with self-synthesized examples. Previous efforts on voice conversion focus on explicitly disentangling speech representations to separately encode speaker characteristics and linguistic content. However, disentangling speech representations to capture such attributes using task-specific loss terms can lead to information loss by discarding finer nuances of the original signal. In this work, instead of explicitly disentangling attributes with loss terms, we present a framework to train a controllable voice conversion model on entangled speech representations derived from self-supervised learning and speaker verification models. First, we develop techniques to derive prosodic information from the audio signal and SSL representations to train predictive submodules in the synthesis model. Next, we propose a training strategy to iteratively improve the synthesis model for voice conversion, by creating a challenging training objective using self-synthesized examples. In this training approach, the current state of the synthesis model is used to generate voice-converted variations of an utterance, which serve as inputs for the reconstruction task, ensuring a continuous and purposeful refinement of the model. We demonstrate that incorporating such self-synthesized examples during training improves the speaker similarity of generated speech as compared to a baseline voice conversion model trained solely on heuristically perturbed inputs. SelfVC is trained without any text and is applicable to a range of tasks such as zero-shot voice conversion, cross-lingual voice conversion, and controllable speech synthesis with pitch and pace modifications. SelfVC achieves state-of-the-art results in zero-shot voice conversion on metrics evaluating naturalness, speaker similarity, and intelligibility of synthesized audio.

<!-- more -->
