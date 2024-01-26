---
title: Unveiling NVIDIA NeMo's Parakeet-TDT -- Turbocharged ASR with Unraveled Accuracy
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

# Unveiling NVIDIA NeMo's Parakeet-TDT -- Turbocharged ASR with Unraveled Accuracy

Earlier this month, we announced [Parakeet](https://huggingface.co/collections/nvidia/parakeet-659711f49d1469e51546e021), a cutting-edge collection of state-of-the-art ASR models built by [NVIDIA's NeMo](https://nvidia.github.io/NeMo/) toolkit, developed jointly with [Suno.ai](http://suno.ai/). Today, we're thrilled to announce the latest addition to the Parakeet family -- [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b). Parakeet TDT achieves unraveled accuracy while running more than 70% faster over our previous best model, making it a great choice for powering speech recognition engine in diverse environments.


The "TDT" in Parakeet-TDT is short for "Token-and-Duration Transducer", a novel sequence modeling architecture developed by NVIDIA and is open-sourced through [NVIDIA's NeMo](https://nvidia.github.io/NeMo/) toolkit. Our research on TDT models, presented in a [paper](https://arxiv.org/abs/2304.06795) at the ICML 2023 conference, showcases the superior speed and recognition accuracy of TDT models compared to conventional Transducers of similar sizes. 

To put things in perspective, our Parakeet-TDT model, boasting around 1.1 billion parameters, outperforms similar-sized Parakeet-RNNT-1.1b in accuracy, as measured as the average performance among 9 benchmarks on the [HuggingFace Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Additionally, it achieves an impressive real-time factor (RTF) of 8.5e-3, which is 75% faster than Parakeet-RNNT-1.1b's RTF of 15e-3. Remarkably, Parakeet-TDT's RTF is approximately 41% faster than Parakeet-RNNT-0.6b, despite the latter having almost half the model size.

## Usage

NVIDIA NeMo can be installed as a pip package as shown below. Cython and PyTorch (2.0 and above) should be installed before attempting to install NeMo Toolkit.

Then simply use:
```bash 
pip install nemo_toolkit['asr']
```

Once installed, you can evaluate a list of audio files as follows:
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
transcript = asr_model.transcribe(["some_audio_file.wav"])
```


## Understanding Token-and-Duration Transducer

Token-and-Duration Transducer (TDT) represents a significant advancement over Transducer models by drastically reducing wasteful computations during the recognition process. To grasp this improvement, let's delve into the workings of a typical Transducer model.

<figure markdown>
  ![RNNTTOPO](rnnt_topo.png)
  <figcaption><b>Figure 1.</b> <i>Transducer Model Architecture</i></figcaption>
</figure>

Transducer models, as illustrated in Figure 1, consist of an encoder, a decoder, and a joiner. During speech recognition, the encoder processes audio signals, extracting crucial information from each frame. The decoder then extracts information from the predicted text, and the joiner combines data from both the encoder and decoder to determine the text token to predict for each frame. Typically a frame covers 40 to 80 milliseconds of audio signal, while on average people speak a word per 400 milliseconds; for those frames that don't add more text to the output, the Transducer predicts a blank symbol. A typical sequence of predictions of a Transducer looks something like,

<code>
&lt;b> &lt;b> &lt;b> NVIDIA &lt;b> &lt;b> &lt;b> &lt;b> is &lt;b> &lt;b>  a &lt;b>  great &lt;b>  place &lt;b> &lt;b> &lt;b>  to work &lt;b> &lt;b> &lt;b> &lt;b> &lt;b> &lt;b> 
</code>

where <code>&lt;b></code> represents the blank symbol, and will be deleted to get the final out of <code>NVIDIA is a great place to work</code>. As we can see, there are many blanks symbols in the output and this means the Transducer model wasted a lot of time on "blank frames" -- frames for which the model predicts blanks which don't contribute to the final output.

<figure markdown>
  ![TDTTOPO](tdt_topo.png)
  <figcaption><b>Figure 2.</b> <i>Token-and-Duration Transducer Model Architecture</i></figcaption>
</figure>


TDT is designed to reduce the wasted computation by intelligently detecting and skipping potential blank frames during model inference. As Figure 2 shows, when a TDT model processes a frame, it simultaneously predicts two things: 

<ol type="1">
<li>probability of next token P<sub>T</sub>(v|t, u): the token that should be predicted at the current frame;</li>
<li>probability of duration P<sub>D</sub>(d|t, u): the number of frames the current token lasts before the model can make the next token prediction. 
</ol>

The TDT model is trained to maximize the number of frames skipped by using the duration prediction while maintaining the same recognition accuracy, thus bring a significant speedup to recognition speed.

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
    * [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)
    * [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
    * [Investigating End-to-End ASR Architectures for Long Form Audio Transcription](https://arxiv.org/abs/2309.09950)
