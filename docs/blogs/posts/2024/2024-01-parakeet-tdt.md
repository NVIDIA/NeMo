---
title: Introducing NVIDIA NeMo's Parakeet-TDT Models : Elevating Speech Recognition to New Heights
author: [Hainan Xu, Nithin Rao Koluguri, Somshubra Majumdar]
author_gh_user: [hainan-xv, nithinraok, titu1994]
readtime: 5
date: 2024-01-25

# Optional: Categories
categories:
  - Announcements

# Optional: OpenGraph metadata
og_title: NVIDIA NeMo Parakeet-TDT
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
description: State of the Art Speech Recognition for Everyone
---

# Introducing NVIDIA NeMo's Parakeet-TDT Models: Elevating Speech Recognition to New Heights!

Earlier this month, we announced Parakeet, a cutting-edge collection of state-of-the-art ASR models built by [NVIDIA's NeMo](https://nvidia.github.io/NeMo/) toolkit. Today, we're thrilled to announce the latest addition to the Parakeet family -- [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b). With improved recognition accuracy over our previous best model and a significantly improved inference speed, Parakeet-TDT elevates speech recognition to a new height.


The "TDT" in Parakeet-TDT is short for "Token-and-Duration Transducer", a novel sequence modeling architecture developed by NVIDIA and is open-sourced through [NVIDIA's NeMo](https://nvidia.github.io/NeMo/) toolkit. We have also published a [paper on TDT models](https://arxiv.org/abs/2304.06795) at the ICML 2023 conference. When compared to conventional Transducers with similar model sizes, TDT models recognize speech significantly faster, and gives superior recognizion accuracies. To put this in context, with around 1.1 billion parameters, our Parakeet-TDT model not only outperforms Parakeet-RNNT-1.1b in terms of accuracy, as measured as the average performance among 9 benchmarks on the [HuggingFace Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) but also acheives an impressive real-time factor (RTF) of 0.0085, which is 75% faster than Parakeet-RNNT-1.1b's RTF of 0.015. This real-time factor is so impressive that it is even around 50% faster than the RTF of 0.012 of Parakeet-RNNT-0.6b, which is almost half the model size of Parakeet-TDT-1.1b.

We thank [Suno.ai](http://suno.ai/) for collaborating on this project.

## TDT model explained

Token-and-Duration Transducer (TDT) is an improvement of Transducer models that got rid of a lot of wasted computations during Transducer computation. To explain that, let's first go over how a Transducer model works.

When a Transducer model recognizes speech, it first uses an encoder to process speech signals and extract the most useful information from each frame. Frame length typically varies from 40 to 80 milliseconds, depending on the sub-sampling used in the encoder. Then the Transducer model will sequentially read those processed frames, and for each frame, either predicts a text token, or a special "blank" symbol, meaning there is nothing to predict for that frame. Note, the average speech rate for humans is around 2.5 words per second, or a word per 400 milliseconds. Say we use 80 millisecond frames, then there are on average 4 blanks for every non-blank token. A typical sequence of predictions of a Transducer looks something like,

\<b\> \<b\> \<b\> \<b\> NVIDIA \<b\> \<b\> \<b\> is \<b\> \<b\> \<b\> a \<b\> \<b\> great \<b\> \<b\> \<b\> \<b\> place \<b\> \<b\> \<b\> \<b\> to \<b\> work \<b\> \<b\> \<b\> \<b\> \<b\> \<b\>   

where \<b\> represents the blank symbol. As we can see, there are many blanks symbols in the output and this means the Transducer model wasted a lot of time on "blank frames" -- frames for which the model predicts blanks and will not contribute to the final output.

TDT is designed to reduce the wasted computation by cleverly detecting and skipping potential blank frames during model inference, thus significantly improve the recognition speed. When a TDT model reads a frame, it simultaneously predicts two things: 1. what is the token that should be predicted at the current frame and 2. what is the next frame the model should jump to in order to predict the next token. For example at the beginning of the sequence above, the TDT model reads the first frame, can predict a blank, and then decide to jump to frame 5 directly and predict the word "NVIDIA"; instead of going through frames 2, 3, 4 and predict 3 consecutive blanks.  This allows the TDT model to singificantly reduce wasted time on blank frames, thus bring a significant speedup to recognition speed.

## Usage

NVIDIA NeMo can be installed as a pip package as shown below. Cython and PyTorch (2.0 and above) should be installed before attempting to install NeMo Toolkit.

Then simply use:
```bash 
pip install nemo_toolkit['asr']
```

Once installed, you can evaluate a list of audio files as follows:
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")
transcript = asr_model.transcribe(["some_audio_file.wav"])
```

## Long-Form Speech Inference

Once you have a Fast Conformer model loaded, you can easily modify the attention type to limited context attention after building the model. You can also apply audio chunking for the subsampling module to perform inference on huge audio files!

!!! note
    These models were trained with global attention, and switching to local attention will degrade their performance. However, they will still be able to transcribe long audio files reasonably well.

For limited context attention on huge files (upto 11 hours on an A100), perform the following steps:

```python
# Enable local attention
asr_model.change_attention_model("rel_pos_local_attn", [128, 128])  # local attn

# Enable chunking for subsampling module
asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

# Transcribe a huge audio file
asr_model.transcribe(["<path to a huge audio file>.wav"])  # 10+ hours !
```

## Additional Resources

* [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
* [HuggingFace ASR Leaderboard Evaluation](https://github.com/huggingface/open_asr_leaderboard)
* [NeMo Parakeet Models on HuggingFace](https://huggingface.co/models?library=nemo&sort=trending&search=parakee)
* [NVIDIA NeMo Webpage](https://github.com/NVIDIA/NeMo)
* [NVIDIA NeMo ASR Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/index.html)
* Papers:
    * [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
    * [Investigating End-to-End ASR Architectures for Long Form Audio Transcription](https://arxiv.org/abs/2309.09950)
