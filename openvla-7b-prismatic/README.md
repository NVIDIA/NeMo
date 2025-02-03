---
library_name: transformers
tags:
- robotics
- vla
- image-text-to-text
- multimodal
- pretraining
license: mit
language:
- en
pipeline_tag: image-text-to-text
---

# OpenVLA 7B (Prismatic-Compatible Version)

<b>This is the same model as the [OpenVLA 7B model](https://huggingface.co/openvla/openvla-7b), except that this checkpoint is in a format that is
compatible with the training script from the original [Prismatic VLMs project codebase](https://github.com/TRI-ML/prismatic-vlms), which the OpenVLA
team built on top of to develop the OpenVLA model. See details for the OpenVLA 7B model here: https://huggingface.co/openvla/openvla-7b</b>

This Prismatic-compatible checkpoint may be useful if you wish to <b>fully fine-tune</b> OpenVLA (all 7.5 billion parameters) via native PyTorch Fully
Sharded Data Parallel (FSDP) using the Prismatic VLMs training script. If you instead wish to do Parameter-Efficient Fine-Tuning via LoRA, you
can use the OpenVLA checkpoint linked above, which is compatible with the Hugging Face `transformers` library. We recommend fine-tuning via LoRA if
you do not have sufficient compute to fully fine-tune a 7B-parameter model (e.g., multiple A100/H100 GPUs).

## Usage Instructions

See the [OpenVLA GitHub README](https://github.com/openvla/openvla/blob/main/README.md) for instructions on how to use this checkpoint for full fine-tuning.

## Citation

**BibTeX:**

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```