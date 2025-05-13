[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### [User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html) | [Tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) | [Technical Blog](https://developer.nvidia.com/blog/tag/nemo/) | [Website](https://www.nvidia.com/en-us/ai-data-science/products/nemo/)

# **NVIDIA NeMo Framework**

- [Introduction](#introduction)
- [Latest News](#latest-news)
- [What's New in NeMo 2.0](#whats-new-in-nemo-20)
- [Features](#features)
- [Install NeMo Framework](#install-nemo-framework)
- [Quickstart](#quickstart)
- [Tutorials](#tutorials)
- [Resources](#resources)
- [Discussions Board](#discussions-board)
- [Contributing](#contributing)
- [Publications](#publications)
- [Licenses](#licenses)

## **Introduction**

NVIDIA NeMo Framework is a comprehensive, scalable, and cloud-native platform designed for AI engineers to develop custom generative AI models. It supports a wide range of applications, including Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-To-Speech (TTS), Vision Language Models (VLMs), and Natural Language Processing (NLP). NeMo enables efficient creation, customization, and deployment of AI models by leveraging pre-trained model checkpoints and existing code. With features like NeMo-Run for streamlined experiment management, NeMo-Curator for data curation, and support for advanced fine-tuning techniques, AI engineers can accelerate their development process and achieve high-performance results.

## **Latest News**

- **Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework**: By using the Hugging Face ecosystem, [NeMo AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework/) enables effortless integration of a vast array of LLMs, without requiring explicit checkpoint rewrites. All models are natively supported, with a subset of the most popular also receiving optimized Megatron-Core support.

- **Turbocharge LLM Training Across Long-Haul Data Center Networks with NVIDIA Nemo Framework**: NeMo Framework 25.02 and NVIDIA Megatron-Core 0.11.0 bring new capabilities for [multi-data center LLM training](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/). This update enables users to scale training beyond the physical and operational limits of a single data center, unlocking unprecedented efficiency and performance by harnessing the combined power of multiple sites.

- **Pretraining and Fine-Tuning Hugging Face Models with AutoModel**: NeMo Framework's latest feature, [NeMo AutoModel](https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/index.html), now supports a wide range of Hugging Face models. Version 25.02 focuses specifically on [AutoModelForCausalLM](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm) for [text generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) tasks, with plans to expand support to additional model families like Vision Language Models in future releases.
- **Training with Blackwell using NeMo Framework**: Blackwell is now supported in NeMo Framework, with version 25.02 emphasizing functional parity for the B200 configuration. Additional optimizations are scheduled for upcoming releases.
- [LLM Model Pruning and Knowledge Distillation with NVIDIA NeMo Framework](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/) (2/12/2025)
- [Improving Translation Quality with Domain-Specific Fine-Tuning and NVIDIA NIM](https://developer.nvidia.com/blog/improving-translation-quality-with-domain-specific-fine-tuning-and-nvidia-nim/) (2/5/2025)
- [Enhancing Generative AI Model Accuracy with NVIDIA NeMo Curator](https://developer.nvidia.com/blog/enhancing-generative-ai-model-accuracy-with-nvidia-nemo-curator/) (1/13/2025)
- [Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform/) (1/9/2025)
- [Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities](https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/) (1/7/2025)
- [Enhance Your Training Data with New NVIDIA NeMo Curator Classifier Models](https://developer.nvidia.com/blog/enhance-your-training-data-with-new-nvidia-nemo-curator-classifier-models/) (12/19/2024)

## **What's New in NeMo 2.0**

- **Python-based Configuration**: NeMo 2.0 shifts from YAML files to [Python-based configuration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html). This change offers more flexibility and better integration with IDEs for code completion and type checking. Users can now leverage Python's full capabilities to create more dynamic and complex configurations, which can be particularly useful for advanced experiments and custom setups.
- **PyTorch Lightning Integration**: By adopting [PyTorch Lightning’s](https://github.com/NVIDIA/NeMo/tree/main/nemo/lightning) modular abstractions, NeMo 2.0 makes it easier to adapt the framework to specific use cases and experiment with various configurations. This integration simplifies the process of writing and maintaining training loops, and it provides out-of-the-box support for multi-GPU and distributed training.
- **NeMo-Run**: This new library simplifies the configuration, execution, and management of machine learning experiments. It integrates with NeMo Framework, enabling efficient model pretraining and fine-tuning across various environments. [NeMo-Run](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/index.html) includes tools and APIs for efficiently managing distributed training, handling data parallelism, and optimizing resource utilization.  
- **Enhanced Models**: NeMo 2.0 includes support for large language models (LLMs) like [Llama 3](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemo-2.0/llms/llama.html), [Mixtral](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/mixtral.html), and [Nemotron](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron.html). It also introduces new models for [Automatic Speech Recognition (ASR)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/intro.html) and [Text-to-Speech (TTS)](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html). These models are designed to leverage the latest advancements in AI research and provide state-of-the-art performance.
- **Custom Tokenizer Training**: Users can now train [custom tokenizers](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/tokenizer.html) within NeMo 2.0. This feature allows for more control over the tokenization process, which can be crucial for domain-specific applications or languages with unique linguistic properties.
- **World Foundation Models:** NeMo Cosmos introduces advanced [Autoregressive](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md) and [Diffusion](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/inference/README.md) WFMs trained on extensive robotics and driving data, as well as advanced video tokenizers and AI-accelerated data pipelines. These enhancements are designed to improve synthetic video creation for autonomous vehicles (AV) and humanoid robots.

## **Features**

NeMo Framework lets you optimize your AI workflow by leveraging the following features:

- **Performance and Scalability**: Accelerates the entire AI workflow from data preparation to model training and inference using techniques like model parallelization and optimized attention mechanisms for high training throughput, supporting execution on-premises, in data centers, or with cloud providers.

  - [Delivers over 800 TFLOPs/sec/GPU for exceptional computational power.](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)
  - [Efficiently trained over clusters with 16+ GPUs for large-scale deployments](https://docs.nvidia.com/nemo-framework/user-guide/24.07/playbooks/autoconfigurator.html)
  - [Supports sequence lengths exceeding 1 million, enabling complex and extensive tasks](https://docs.nvidia.com/nemo-framework/user-guide/latest/longcontext/index.html)
  - [Employs 4D parallelism to optimize model training and inference.](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html)
  - [Utilizes GPU-accelerated data curation for faster and more efficient data preparation](https://docs.nvidia.com/nemo-framework/user-guide/24.07/datacuration/index.html)

- **Model Coverage**: Provides end-to-end support for developing LLMs and MMs, including tools for data curation, training, and customization, and supports 23 model families including:

  - [Large Language Models (LLMs)](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html)
  - [Speech Synthesis Models (SSMs)](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
  - [Mixture-of-Experts (MOEs)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/moe.html)
  - [Speech Denoising (SD)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/audio/intro.html)
  - [Vision-Language Models (VLMs)](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/index.html)
  - [Waveform Models (WFMs)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/tts/models.html)
  - [Automatic Speech Recognition (ASR) Models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
  - [Text-to-Speech (TTS) Models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html)
  - [Natural Language Processing (NLP) Models](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemonlpmodels)
  - [Synthetic Data Generation (SDG) Models](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/syntheticdata.html)

- **State-of-the-Art Algorithms**: Supports state-of-the-art algorithms, including advanced fine-tuning and model alignment techniques, ensuring high performance and accuracy in AI applications.

  - [**Parameter Efficient Fine-Tuning (PEFT)**](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/gpt/peft.html):
    - [Low-Rank Adaptation (LoRA)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/supported_methods.html)
    - [Quantized Low-Rank Adaptation (QLoRA)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/qlora.html)
  - [**Model Alignment**:](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html)
    - [Reinforcement Learning from Human Feedback (RLHF)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rlhf.html)
    - [Proximal Policy Optimization (PPO)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rlhf.html)
    - [Direct Policy Optimization (DPO)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html)
    - [Knowledge Transfer Optimization (KTO)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/model-optimization/distillation/distillation.html)
    - [Iterative Policy Optimization (IPO)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html)
    - [Reinforcement Learning with Artificial Intelligence Feedback (RLAIF)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/modelalignment/cai.html)
    - [Steerable Language Models (SteerLM)](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html)
    - [Rejection Sampling](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rs.html)

- **Usability**: Offers an intuitive interface for constructing comprehensive workflows, making it easy to manage experiments across different environments.

  - [Hugging Face-like Pythonic APIs](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemo-2.0/features/hf-integration.html)
  - [Intuitive interface for constructing comprehensive workflows](https://docs.nvidia.com/nemo-framework/user-guide/24.07/overview.html)
  - [NeMo-Curator for efficient data curation](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html)
  - [NeMo-Run for streamlined configuration and execution](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/index.html)
  - [Extensive documentation and user guides for ease of use](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
  - [Pre-trained model checkpoints for quick deployment](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/checkpoints/intro.html)  
  - [Jupyter Notebook support for interactive development](https://docs.nvidia.com/nemo-framework/user-guide/latest/why-nemo.html)

- **Compatibility**: Ensures seamless integration with diverse computing environments, such as [SLURM and Kubernetes](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemorun/guides/execution.html).

  - [Fault tolerance to ensure smooth training](https://docs.nvidia.com/nemo-framework/user-guide/latest/resiliency.html)
  - [Supports both SFT and PEFT techniques](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/index.html)
  - [Works with popular machine learning frameworks like TensorFlow and PyTorch](https://docs.nvidia.com/nemo-framework/user-guide/24.07/getting-started.html)
  - [Integrates with cloud service providers like AWS, Azure, and Google Cloud](https://docs.nvidia.com/nemo-framework/user-guide/24.07/cloudserviceproviders.html)
  - [Cross-platform support for different operating systems (Linux, Windows, macOS)](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
  - [Docker container support for easy deployment and scalability](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)

## **Install NeMo Framework**

Several options are available for installing NeMo Framework:

- Docker container  
- Conda and Pip

For more information, please see [Install NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.12/installation.html) in the NeMo Framework User Guide.

## **Quickstart**

Check out the [Quickstart with NeMo-Run](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples on how to use NeMo-Run to launch NeMo 2.0 experiments locally and on a Slurm cluster. To run a simple training loop using the train API from the LLM collection, see [Quickstart with NeMo 2.0 API](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).

## **Tutorials**

The best way to get started with NeMo is to try out our tutorials. These tutorials cover various NeMo collections and provide both introductory and advanced topics.

| **Collection/Tutorial** | **Description** |
|--------------|-----------------|
| **[Automatic Speech Recognition](https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr)** | Build and fine-tune ASR models using the NeMo Framework, covering basic and advanced training, model evaluation, streaming and offline ASR, and voice activity detection. |
| [ASR\_with\_NeMo](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb) | Provides a foundational understanding of ASR concepts and their implementation using the NeMo Framework. It introduces the basics of ASR, including the generative and end-to-end models, and demonstrates how to construct and train an end-to-end ASR pipeline. |
| [Online\_ASR\_Microphone\_Demo\_Cache\_Aware\_Streaming](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb) | Demonstrates real-time (streaming) speech recognition using audio recorded from your microphone. It also explains how to use a NeMo chunk-aware FastConformer model with caching enabled. |
| [ASR\_with\_Adapters](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/asr_adapters/ASR_with_Adapters.ipynb) | Introduces adapters and their use cases with ASR models. It explores domain adaptation of a pre-trained model using adapter modules, discusses the general advantages and disadvantages of adapters, and trains a model to adapt to a toy dataset. |
| [Speech\_Commands](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Speech_Commands.ipynb) | Focuses on the task of speech classification, a subset of speech recognition. It involves classifying a spoken sentence into a single label, enabling the model to recognize and perform actions based on spoken commands. |
| **[Audio Processing](https://github.com/NVIDIA/NeMo/tree/main/tutorials/audio)** | Set up and train a simple single-channel speech enhancement model in NeMo, including data preparation and model configuration for both single-output and dual-output models. |
| [Speech\_Enhancement\_with\_NeMo](https://github.com/NVIDIA/NeMo/blob/main/tutorials/audio/speech_enhancement/Speech_Enhancement_with_NeMo.ipynb) | Demonstrates the basic steps required to set up and train a simple single-channel speech enhancement model using NeMo. |
| [Speech\_Enhancement\_with\_Online\_Augmentation](https://github.com/NVIDIA/NeMo/blob/main/tutorials/audio/speech_enhancement/Speech_Enhancement_with_Online_Augmentation.ipynb) | Illustrates the basic steps required to set up and train a simple single-channel speech enhancement model in NeMo using online augmentation with noise and room impulse response (RIR). |
| **[Large Language Models](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm)** | Build, train, and fine-tune LLMs using the NeMo Framework, encompassing model training, fine-tuning, optimization techniques, reinforcement learning, and the development of multimodal models. |
| [nemo2-peft](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama/nemo2-sft-peft/nemo2-peft.ipynb) | Demonstrates how to customize foundation models to enhance their performance on specific tasks using NeMo 2.0. |
| [llama3-lora-nemofw](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama/biomedical-qa/llama3-lora-nemofw.ipynb) | Shows how to perform LoRA PEFT Llama 3 8B on PubMedQA using the NeMo Framework. PubMedQA is a Question-Answering dataset for biomedical texts. |
| [Data\_pipeline.ipynb](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama/slimpajama/data_pipeline.ipynb) | Demonstrates how to transform a raw pretraining dataset into a configured data module for pretraining with a NeMo 2.0 recipe. The SlimPajama-627B dataset is used as a reference. |
| **[Multimodal Models](https://github.com/NVIDIA/NeMo/tree/main/tutorials/multimodal)** | Build, train, and optimize MMs that integrate various data types (text, images, and audio), focusing on data preparation, model training, and optimization techniques. |
| [NeVA Mixtral](https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/NeVA%20Mixtral%20Tutorial.ipynb) | Illustrates the new features within Neural Video Assistant (NeVA), such as Mistral and Mixtral Support, Token Compression, and SigLIP support. |
| [Prompt Formatter](https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/Prompt%20Formatter%20Tutorial.ipynb) | Introduces NeMo's PromptFormatter API available in module nemo.collections.common.prompts. After finishing this tutorial you will be familiar with the existing prompt formatters, how to use them, and how to build your own. |
| [SDXL](https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/SDXL%20Tutorial.ipynb) | Illustrates how to train and perform inference using Stable Diffusion XL with the NeMo toolkit. |
| **[Text-to-Speech](https://github.com/NVIDIA/NeMo/tree/main/tutorials/tts)** | Build, train, and fine-tune TTS models, including training, fine-tuning for different languages and voices, and performance optimization. |
| [Audio\_Codec\_Training](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Training.ipynb) | Demonstrates how to train and fine-tune neural audio codecs. Neural audio codecs are deep learning models that compress audio into a low bitrate representation, useful for tasks like TTS and ASR. |
| [Tacotron2\_Training](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Tacotron2_Training.ipynb) | Shows how to train the Tacotron2 model as part of the TTS pipeline. It includes an introduction to the Tacotron2 model, instructions on training Tacotron2 using the LJSpeech dataset, and guidelines on collecting audio data to train Tacotron2 for different voices and languages using custom datasets. |
| [VITS and NeMo](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Vits_Training.ipynb) | Demonstrates how to train the Variational Inference Text-to-Speech (VITS) model as part of the TTS pipeline. It includes an introduction to the VITS model and detailed instructions on training VITS using the LJSpeech dataset. |
| **[Natural Language Processing](https://github.com/NVIDIA/NeMo/tree/main/tutorials/nlp)** | Build, train, and fine-tune NLP models using the NeMo Framework, covering model training, fine-tuning, optimization, data preprocessing, and evaluation. |
| [Punctuation\_and\_Capitalization\_Lexical\_Audio](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization_Lexical_Audio.ipynb) | Demonstrates how to train a model to predict punctuation and capitalization for ASR outputs using both text and audio inputs. The goal is to improve the performance of downstream tasks such as named entity recognition and machine translation. |
| [Token\_Classification\_Named\_Entity\_Recognition](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb) | Shows how to perform Named Entity Recognition (NER), which involves detecting and classifying entities in text. It uses the Groningen Meaning Bank (GMB) corpus for training and provides instructions for downloading, preprocessing, and converting the dataset into the required format. |
| **[Speaker Tasks](https://github.com/NVIDIA/NeMo/tree/main/tutorials/speaker_tasks)** | Build, train, and fine-tune models for speaker diarization, identification, and verification, including voice activity detection, speaker embedding extraction, and clustering. |
| [End\_to\_End\_Diarization\_Inference](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/End_to_End_Diarization_Inference.ipynb) | Covers downloading a sample audio file, visualizing speaker activity, performing diarization with the Sortformer model, and post-processing the results to optimize Diarization Error Rate (DER). |
| [Speaker\_Identification\_Verification](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb) | Demonstrates how to set up and train a TitaNet model for speaker recognition, including how to fine-tune a pre-trained model, save and restore checkpoints, and extract speaker embeddings from audio files. |
| **[Tools](https://github.com/NVIDIA/NeMo/tree/main/tutorials/tools)** | Utilize various tools to enhance AI model development, such as data simulation, forced alignment, and segmentation, to improve the efficiency and accuracy of your models. |
| [SDE\_HowTo\_v2](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/SDE_HowTo_v2.ipynb) | Demonstrates how to use Speech Data Explorer (SDE) in Comparison mode to evaluate two ASR models on a given test set and identify differences in their predictions. |
| [NeMo\_Forced\_Aligner\_Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb) | Uses NeMo Forced Aligner to generate token and word alignments for Neil Armstrong's moon landing video. The Advanced SubStation Alpha-format subtitle files will add subtitles with token-by-token and word-by-word highlighting. |

## **Resources**

- **NeMo Framework User Guide**: For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-Framework/user-guide/latest/nemo-2.0/index.html).
- **Feature Guide**: For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-Framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
- **Migration Guide**: To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-Framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.
- **Pretrained NeMo Models**: Our pretrained models are freely available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
- **Recipes**: For additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run, see the [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes).
- **Examples**: For advanced users looking to leverage the full capabilities of NeMo Framework for their AI/machine learning projects, we offer a collection of Python [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples). These scripts support multi-GPU/multi-node training.
- **Releases**: For the latest information on software component versions, changelogs, and known issues, see [Releases](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html).

## **Discussions Board**

Check out our [Discussions board](https://github.com/NVIDIA/NeMo/discussions) where you can ask questions and start discussions.

## **Contributing**

We welcome community contributions\! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.

## **Publications**

Explore our growing list of [publications](https://nvidia.github.io/NeMo/publications/) utilizing the NeMo Framework.

To contribute an article, please submit a pull request to the `gh-pages-src` branch. Detailed information is available in the README located at the `gh-pages-src` branch.

## **Licenses**

- [NeMo GitHub Apache 2.0 license](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme)  
- NeMo is licensed under the [NVIDIA AI PRODUCT AGREEMENT](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/). By pulling and using the container, you accept the terms and conditions of this license.
