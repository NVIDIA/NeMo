---
title: Announcing NVIDIA NeMo Parakeet ASR Models for Pushing the Boundaries of Speech Recognition
author: [Nithin Rao Koluguri, Somshubra Majumdar]
author_gh_user: [nithinraok, titu1994]
readtime: 5
date: 2024-01-03

# Optional: Categories
categories:
  - Announcements

# Optional: OpenGraph metadata
og_title: NVIDIA NeMo Parakeet
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
description: State of the Art Speech Recognition for Everyone
---

# Announcing NVIDIA NeMo Parakeet ASR Models for Pushing the Boundaries of Speech Recognition

[NVIDIA NeMo](https://nvidia.github.io/NeMo/), a leading open-source toolkit for conversational AI, announces the release of Parakeet, a family of state-of-the-art automatic speech recognition (ASR) models, capable of transcribing spoken English with exceptional accuracy. Developed in collaboration with Suno.ai, Parakeet ASR models mark a significant leap forward in speech recognition, paving the way for more natural and efficient human-computer interactions.

NVIDIA announces four Parakeet models based on RNN Transducer / Connectionist Temporal Classification decoders and the size of the models. They boast 0.6-1.1 billion parameters abd capable of tackling diverse audio environments. Trained on only a 64,000-hour dataset encompassing various accents, domains, and noise conditions, the model delivers exceptional word error rate (WER) performance across benchmark datasets, outperforming previous models.

* [Parakeet RNNT 1.1B](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) - Best recognition accuracy, modest inference speed. Best used when the most accurate transcriptions are necessary.
* [Parakeet CTC 1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b) - Fast inference, strong recognition accuracy. A great middle ground between accuracy and speed of inference.
* [Parakeet RNNT 0.6B](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) - Strong recognition accuracy, accurate, fast inference. Useful for large-scale inference on limited resources.
* [Parakeet CTC 0.6B](https://huggingface.co/nvidia/parakeet-ctc-0.6b) - Fastest speed, modest recognition accuracy. Useful when transcription speed is the most important.

<!-- more -->

Parakeet models exhibit resilience against non-speech segments, including music and silence, effectively preventing the generation of hallucinated transcripts.

Built using the [NVIDIA NeMo toolkit](https://github.com/NVIDIA/NeMo), Parakeet prioritizes user-friendliness and flexibility. With pre-trained checkpoints readily available, integrating the model into your projects is a breeze. Whether looking for immediate inference capabilities or fine-tuning for specific tasks, NeMo provides a robust and intuitive framework to leverage the model's full potential.

Key benefits of Parakeet models:

* **State-of-the-art accuracy:** Superior WER performance across diverse audio sources and domains with strong robustness to non-speech segments.
* **Different model sizes:** Two models with 0.6B and 1.1B parameters for robust comprehension of complex speech patterns.
* **Open-source and extensibility:** Built on NVIDIA NeMo, allowing for seamless integration and customization.
* **Pre-trained checkpoints:** Ready-to-use models for inference or fine-tuning.
* **Permissive license:** Released under CC-BY-4.0 license, model checkpoints can be used in any commercial application.

Parakeet is a major step forward in the evolution of conversational AI. Its exceptional accuracy, coupled with the flexibility and ease of use offered by NeMo, empowers developers to create more natural and intuitive voice-powered applications. The possibilities are endless, from enhancing the accuracy of virtual assistants to enabling seamless real-time communication.. 

The Parakeet family of models achieves state-of-the-art numbers on the [HuggingFace Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Users can try out the [parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) firsthand at the [Gradio demo](https://huggingface.co/spaces/nvidia/parakeet-rnnt-1.1b). To access the model locally and explore the toolkit, visit the [NVIDIA NeMo Github page](https://github.com/NVIDIA/NeMo). 

## Architecture Details

Parakeet models are based on the [Fast Conformer architecture published in ASRU 2023](https://arxiv.org/abs/2305.05084). Fast Conformer is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling, modified Convolution kernel size, and an efficient subsampling module. Additionally it supports inference on very long audio segments (up to 11 hours of speech) on an A100 80GB card using Local Attention. The model is trained end-to-end using the Transducer decoder (RNNT) or Connectionist Temporal Classification decoder. For further details on long audio inference, please refer to the ICASSP 2024 paper “[Investigating End-to-End ASR Architectures for Long Form Audio Transcription](https://arxiv.org/abs/2309.09950)”.

<figure markdown>
  ![Parakeet architecture](https://github.com/NVIDIA/NeMo/releases/download/v1.21.0/asset-post-2024-01-03-parakeet_arch.png)
  <figcaption><b>Figure 1.</b> <i> Fast Conformer Architecture shows blocks of downsampling, conformer encoder blocks with limited context attention (LCA), and global token (GT).</i></figcaption>
</figure>

## Usage

NVIDIA NeMo can be installed as a pip package as shown below. Cython and PyTorch (2.0 and above) should be installed before attempting to install NeMo Toolkit.

Then simply use 
```bash 
pip install nemo_toolkit['asr']
```

Once installed, you can evaluate a list of audio files as follows:
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")
transcript = asr_model.transcribe(["some_audio_file.wav"])
```

## Additional Resources

* [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
* [NeMo Parakeet Models on HuggingFace](https://huggingface.co/models?library=nemo&sort=trending&search=parakee)
* [NVIDIA NeMo Webpage](https://github.com/NVIDIA/NeMo)
* [NVIDIA NeMo ASR Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/index.html)
* Papers:
    * [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
    * [Investigating End-to-End ASR Architectures for Long Form Audio Transcription](https://arxiv.org/abs/2309.09950)

## About NVIDIA NeMo

[NVIDIA NeMo](https://nvidia.github.io/NeMo/) is an open-source toolkit for building state-of-the-art conversational AI models. It provides a collection of pre-built modules for speech recognition, natural language processing, and text-to-speech, allowing developers to quickly and easily create custom conversational AI applications. 
