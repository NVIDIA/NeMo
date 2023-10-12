---
title: "SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF"
author: [Yi Dong, Zhilin Wang, Makesh Narsimhan Sreedhar, Xianchao Wu, Oleksii Kuchaiev]
author_gh_user: []
readtime: 30
date: 2023-10-09

# Optional: Categories
categories: [Natural Language Processing]
continue_url: https://arxiv.org/abs/2310.05344

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://arxiv.org/abs/2310.05344)

Model alignment with human preferences is an essential step in making Large Language Models (LLMs) helpful and consistent with human values. It typically consists of supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) stages. However, RLHF faces inherent limitations stemming from a complex training setup and its tendency to align the model with implicit values that end users cannot control at run-time. Moreover, reward models in RLHF stage commonly rely on single-dimensional feedback as opposed to explicit, multifaceted signals that indicate attributes such as helpfulness, humor, and toxicity. To address these limitations, we propose SteerLM, a supervised fine-tuning method that empowers end-users to control responses during inference. SteerLM conditions responses to conform to an explicitly defined multi-dimensional set of attributes, thereby empowering a steerable AI capable of generating helpful and high-quality responses while maintaining customizability. Experiments show that SteerLM trained on open source datasets generates responses that are preferred by human and automatic evaluators to many state-of-the-art baselines trained with RLHF while being much easier to train. Try SteerLM at https://huggingface.co/nvidia/SteerLM-llama2-13B

<!-- more -->

