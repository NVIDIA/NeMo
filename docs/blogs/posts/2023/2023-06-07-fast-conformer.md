---
title: Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition
author: [Dima Rekesh, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Ankur Kumar, Boris Ginsburg]
author_gh_user: [bmwshop, sam1373, titu1994, VahidooX, stevehuang52, AlexGrinch, iankur, borisgin]
date: 2023-06-07
readtime: 10
categories:
- Papers

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
og_image: https://github.com/NVIDIA/NeMo/releases/download/v1.18.0/asset-post-fast-conformer-diagram.png
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/)
description: Efficient training and inference on long audio with Fast Conformer architecture
---

# Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition

The Conformer architecture, introduced by [Gulati et al.](https://arxiv.org/abs/2005.08100) has been a standard architecture used for not only Automatic Speech Recognition, but has also been extended to other tasks such as Spoken Language Understanding, Speech Translation, and used as a backbone for Self Supervised Learning for various downstream tasks. While they are highly accurate models on each of these tasks, and can be extended for use in other tasks, they are also very computationally expensive. This is due to the quadratic complexity of the attention mechanism, which makes it difficult to train and infer on long sequences, which are used as input to these models due to the granular stride of audio pre-processors (commonly Mel Spectrograms or even raw audio signal in certain models with 10 milliseconds stride). Furthermore, the memory requirement of quadratic attention also significantly limits the audio duration during inference.

<!-- more -->

In this paper, we introduce the **Fast Conformer** architecture, which applies simple changes to the architecture to significantly reduce the computational cost of training and inference, all while mantaining the strong results of the original Conformer model. We further show that by modifying (on the fly) the global attention module to a linearly scalable attention mechanism - the same model can be used to train (or finetune) and then infer on long sequences (up to 1 hour !).

Please refer to the paper here: [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

<div align="center">
<img src="https://github.com/NVIDIA/NeMo/releases/download/v1.18.0/asset-post-fast-conformer-diagram.png">
</div>

## Fast Conformer: Architecture

We propose 4 changes to the original Conformer architecture to make it more efficient:

1) **Downsampling module**: The original Conformer paper uses a stack of 2-D Convolutions with a large number of output filters to perform the downsampling in order to reduce the resolution of the incoming audio frames from 10 ms to 40 ms, which makes it more tractable for the subsequent attention layers to operate on. However, these convolutions amount to roughly 20 % of the entire time required to perform a single forward pass of the "Large" Conformer (with 120 M parameters). Furthermore, due to the quadratic attention cost, we can obtain a 4x reduction in all subsequent attention layers by downsampling the audio to 80 ms frames. So as the first change, we perform 8x downsampling before any of the Conformer layers are applied. This reduces GMACs to roughly 65% of the baseline Conformer.

2) **Depthwise convolution**: Multiple other works have shown that it is not necessary to use full Convolution layers and that we can save both compute and memory simply by using Depthwise Separable Convoltions. Therefore, we replace all Convolution layers in the preprocessor by Depthwise Seperable Convolutions. This reduces GMACs to roughly 37% of the baseline Conformer.

3) **Channel reduction**: In literature, it is common for the hidden dimension of the downsampling module to match the `d_model` of the Conformer module for easy application to the subsequent stack of Conformer modules. However, this is not necessary, and we can reduce the number of channels in the downsampling module to reduce the number of parameters and GMACs. We reduce the number of channels to 256, which reduces GMACs to roughly 33% of the baseline Conformer.

4) **Kernel size reduction**: Finally, as we have performed 8x downsampling of the incoming audio, it is no longer required to use the rather large kernel size of 31 in the Convolution layers of the Conformer block. We find that we can roughly reduce it to 9, which preserves the same accuracy while executing slightly faster and reducing the memory cost. This finally reduces GMACs to roughly 33% of the baseline Conformer.


*Below, we tabulate the accuracy vs speed for each component of Fast Conformer modification schema. Models were tested on LibriSpeech test-other incrementally for each modification starting from the original Conformer. Encoder inference speed (samples/sec) was measured with batch size 128 on 20 sec audio samples. The number of parameters (M) is shown for the encoder only.* 

| Encoder             	| WER, Test Other % 	| Inference Samples / sec 	| Params (M) 	| GMACS 	|
|---------------------	|-------------------	|-------------------------	|------------	|-------	|
| Baseline Conformer  	| 5.19              	| 169                     	| 115        	| 143.2 	|
| \  +8X Stride       	| 5.11              	| 303                     	| 115        	| 92.5  	|
| \ \ +Depthwise conv 	| 5.12              	| 344                     	| 111        	| 53.2  	|
| \ \ \ +256 channels 	| 5.09              	| 397                     	| 109        	| 48.8  	|
| \ \ \ \ +Kernel 9   	| 4.99              	| 467                     	| 109        	| 48.7  	|


## Fast Conformer: Linearly Scalable Attention

On an NVIDIA A100 GPU with 80 GB of memory, we find that Conformer reaches the memory limit at around 10-minute long single audio clip. This mean that it is not feasible to perform inference without performing streaming inference which may lead to degraded results.

Fast Conformer, due to its 8x stride fairs a little better and can perform inference on roughly 18-minute long audio clips. However, this is still not sufficient for many use cases.

We therefore extend [Longformer](https://arxiv.org/abs/2004.05150) to the Conformer architecture. Longformer uses local attention augmented with global tokens.  We use a single global attention token, which attends to all other tokens and has all other tokens attend to it, using a separate set of query, key and value linear projections, while others attend in a fixed-size window surrounding each token (see below). 

<div align="center">
<img src="https://github.com/NVIDIA/NeMo/releases/download/v1.18.0/asset-post-fast-conformer-local-attn.png" width="100%">
</div>

By switching to limited context attention, we extend the maximum duration that the model can process at once on a single A100 GPU by ~4x: from 10 minutes for Conformer to 18 minutes for Fast Conformer. Furthermore, you can use a pre-trained Fast Conformer model and readily convert its attention to Longformer attention without any further training ! While this will not use the global attention token, it will still be able to process 70-minute long audio clips.`

## Checkpoints

We provide checkpoints for multiple languages on [HuggingFace](https://huggingface.co/models?sort=downloads&search=fastconformer) and will add more when we support other languages or tasks.

Each of these models are "Punctuation and Capitalization" enabled - meaning that they can be used to perform ASR and Punctuation and Capitalization (PnC) in a single pass and can produce text that is more natural to read. Post-processing to normalize text will be provided in a future release.

Some languages we currently support for ASR are :

- [English](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_pc)
- [French](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc)
- [German](https://huggingface.co/nvidia/stt_de_fastconformer_hybrid_large_pc)
- [Italian](https://huggingface.co/nvidia/stt_it_fastconformer_hybrid_large_pc)
- [Spanish](https://huggingface.co/nvidia/stt_es_fastconformer_hybrid_large_pc)
- [Belarusian](https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc)
- [Croatian](https://huggingface.co/nvidia/stt_hr_fastconformer_hybrid_large_pc)
- [Polish](https://huggingface.co/nvidia/stt_pl_fastconformer_hybrid_large_pc)
- [Ukranian](https://huggingface.co/nvidia/stt_ua_fastconformer_hybrid_large_pc)
- [Russian](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc)

## Data Processing

When constructing datasets with support for Punctuation and Capitalization, dataset preparation is an important piece of the puzzle. When training a model with Punctuation and Capitalization, you may face the following issues:

1) During training, there may be the case the standard evaluation benchmarks do not have punctuation or capitalization, but the model still predicts them, providing an incorrect evaluation of model training. 

2) Not all training data may have punctuation or capitalization, so you may want to filter out such samples to prevent confusing the model about whether it should predict punctuation or capitalization.

In order to provide a consistent and reproducible way to process such dataset, we will begin providing the dataset preprocessing strategies in [Speech Data Processor](https://github.com/NVIDIA/NeMo-speech-data-processor).

Speech Dataset Processor currently hosts dataset processing recipies for Spanish, and we will add more languages in the future.

# Usage

Fast Conformer can be instantiated and used with just a few lines of code when the NeMo ASR library is installed.

## Global Attention

For global attention on modest files (upto 15-18 minutes on an A100), you can perform the following steps :

```python
from nemo.collections.asr.models import ASRModel
  
model = ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_pc")
model.transcribe(["<path to a audio file>.wav"])  # ~10-15 minutes long!

```

## Local Attention

Coming in NeMo 1.20, you can easily modify the attention type to local attention after building the model. Then you can also apply audio chunking for the subsampling module to perform inference on huge audio files!

For local attention on huge files (upto 11 hours on an A100), you can perform the following steps :

```python
from nemo.collections.asr.models import ASRModel
  
model = ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_pc")

model.change_attention_model("rel_pos_local_attn", [128, 128])  # local attn

# (Coming in NeMo 1.20)
model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

model.transcribe(["<path to a huge audio file>.wav"])  # 10+ hours !

```

# Results

By performing the simple modifications in Fast Conformer, we obtain strong scores throughout multiple speech tasks as shown below, all while having more efficient training and inference.

For ASR alone, we obtain a 2.8x speedup as compared to Conformer encoder of similar size for inference, and can use larger batch sizes during training to further speedup training. We also compare results against tasks such as Speech Translation and Spoken Language Understanding in order to validate that these changes lead only to improvements in efficiency and do not degrade accuracy on downstream tasks.

## ASR Results

Below, we list some of our results on Fast Conformer on LibriSpeech test-other. We compare against the original Conformer and other recent efficient architectures. We compare against the [Efficient Conformer from Burchi2021](https://arxiv.org/abs/2109.01163) which uses a progressive downsampling of the Conformer architecture. We also compare against [Kim2022 SqueezeFormer](https://arxiv.org/abs/2206.00888) which uses a U-Net like architecture to progressively downsample the input and upsample it to 40 ms resolution prior to applying the decoder. 

We find that Fast Conformer is able to achieve the same accuracy as the Conformer while being 2.8x faster and using fewer parameters. 

*Fast Conformer-Large with CTC and RNNT decoders trained on Librispeech. Greedy WER (%) was measured on Librispeech test-other.  The number of parameters  (M) and compute (Multiply-Acc, GMAC) are shown for encoder only.*

| Encoder                     	 |   WER, %   	| Params, 	| Compute, 	|
|-------------------------------|:----------:	|:-------:	|:--------:	|
| 	                             | test-other 	|    M    	|   GMACS  	|
| **RNNT decoder**        	     |            	|         	|          	|
| Conformer                   	 |    5.19    	|   115   	|   143.2  	|
| Fast Conformer              	 |    4.99    	|   109   	|   48.7   	|
| **CTC decoder**         	     |            	|         	|          	|
| Conformer                   	 |    5.74    	|   121   	|   149.2  	|
| Eff. Conformer [Burchi2021] 	 |    5.79    	|   125   	|   101.3  	|
| SqeezeFormer [Kim2022]      	 |    6.05    	|   125   	|   91.0   	|
| Fast Conformer              	 |    6.00    	|   115   	|   51.5   	|

## Speech Translation Results

We also evaluate Fast Conformer on the IWSLT 2014 German-English speech translation task. We find that Fast Conformer is able to achieve the same accuracy as the Conformer while being upto 1.8x faster and using fewer parameters.

Our models have been trained on all available datasets from IWSLT22 competition which corresponds to 4k hours of speech. Some of the datasets did not contain German translations, so we generated them ourselves with text-to-text machine translation model trained on WMT21 and in-domain finetuned on Must-C v2.

*Speech Translation, MUST-C V2 tst-COMMON dataset. SacreBLEU, total inference time, and relative inference speed-up were measured with a batch size $32$ for two speech translation models with Conformer-based encoder and either RNNT, or Transformer decoder.*

| Encoder             	     | BLEU 	| Time (sec) 	 | Speed-up 	|
|---------------------------|:----:	|:------------:|:--------:	|
| **Transformer decoder** 	 |      	|      	       |          	|
| Conformer           	     | 31.0 	|   267    	   |    1X    	|
| Fast Conformer      	     | 31.4 	|   161    	   |   1.66X  	|
| **RNNT decoder**    	     |      	|      	       |          	|
| Conformer           	     | 23.2 	|   83    	    |    1X    	|
| Fast Conformer      	     | 27.9 	|   45    	    |   1.84X  	|

## Spoken Language Understanding Results

For the Speech Intent Clasification and Slot Filling task, experiments are conducted using the [SLURP](https://arxiv.org/abs/2011.13205) dataset, where intent accuracy and SLURP-F1 are used as the evaluation metric. For a fair comparison, both Conformer and Fast Conformer models are initialized by training on the same NeMo ASR Set dataset (roughly 25,000 hours of speech) and then the weights of the entire model are finetuned with the respective decoders.

*Speech intent classification and slot filling on SLURP dataset. ESPNet-SLU  and SpeechBrain-SLU models use a [HuBERT](https://arxiv.org/abs/2106.07447) encoder pre-trained via a self-supervised objective on [LibriLight-60k](https://arxiv.org/abs/1912.07875). 
Inference time and relative speed-up against Conformer are measured with batch size 32.*

| Model                                        	     | Intent Acc 	| SLURP F1 	| Inference, sec 	| Rel. Speed-up 	|
|----------------------------------------------------|:----------:	|:--------:	|:--------------:	|:-------------:	|
| SpeechBrain-SLU                              	     |    87.70   	|   76.19  	|        -       	|       -       	|
| ESPnet-SLU                                   	     |    86.52   	|   76.91  	|        -       	|       -       	|
| **Conformer/Fast Conformer+Transformer Decoder** 	 |            	|          	|                	|               	|
| Conformer                                    	     |    90.14   	|   82.27  	|       210      	|       1X      	|
| Fast Conformer                               	     |    90.68   	|   82.04  	|       191      	|      1.1X     	|

## Long Form Speech Recognition Results

While Fast Conformer can be modified post training to do simple inference on long audio, due to the mismatch in attention window with limited future information, the model's WER may degrade a small amount. Users can therefore add global token followed by subsequent fine-tuning for some small steps on the same dataset in order to significantly recover (and outperform) the original models WER.

Note that with Longformer style attention, we can still perform buffered inference with large chunk size - upto 1 hour long, therefore inference on even longer audio can be done efficiently with few inference steps.

*Fast Conformer versus Conformer on long audio. We evaluated four versions of FastConformer: (1) FC with global attention (2) FC with limited context (3) FC with limited context and global token (4) FC with limited context and global token, trained on long concatenated utterances. Models have been evaluated on two long-audio bencmarks: TED-LIUM v3 and Earning 21. Greedy WER(%).*

| Model                     	| TED-LIUM v3 	| Earnings21 	|
|---------------------------	|:-----------:	|:----------:	|
| Conformer                 	|     9.71    	|    24.34   	|
| Fast Conformer            	|     9.85    	|    23.84   	|
| \ + Limited Context       	|     9.92    	|    28.42   	|
| \ \ + Global Token        	|     8.00    	|    20.68   	|
| \ \ \ + Concat utterances 	|     7.85    	|    19.52   	|
