.. _tutorials:

Tutorials
=========

The best way to get started with NeMo is to start with one of our tutorials. These tutorials cover various domains and provide both introductory and advanced topics. They are designed to help you understand and use the NeMo toolkit effectively.

Running Tutorials on Colab
--------------------------

Most NeMo tutorials can be run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.

To run a tutorial:

1. Click the **Colab** link associated with the tutorial you are interested in from the table below.
2. Once in Colab, connect to an instance with a GPU by clicking **Runtime** > **Change runtime type** and selecting **GPU** as the hardware accelerator.

Tutorial Overview
-----------------

.. list-table:: **General Tutorials**
   :widths: 15 25 60
   :header-rows: 1

   * - Domain
     - Title
     - GitHub URL
   * - General
     - Getting Started: NeMo Fundamentals
     - `NeMo Fundamentals <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`_
   * - General
     - Getting Started: Audio translator example
     - `Audio translator example <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/AudioTranslationSample.ipynb>`_
   * - General
     - Getting Started: Voice swap example
     - `Voice swap example <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/VoiceSwapSample.ipynb>`_
   * - General
     - Getting Started: NeMo Models
     - `NeMo Models <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`_
   * - General
     - Getting Started: NeMo Adapters
     - `NeMo Adapters <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/02_NeMo_Adapters.ipynb>`_
   * - General
     - Getting Started: NeMo Models on Hugging Face Hub
     - `NeMo Models on HF Hub <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/Publish_NeMo_Model_On_Hugging_Face_Hub.ipynb>`_

Multimodal Tutorials
~~~~~~~~~~~~~~~~~~~~

* Preparations and Advanced Applications
  - `Multimodal Data Preparation <https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/Multimodal%20Data%20Preparation.ipynb>`_
  - `NeVA (LLaVA) Tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/NeVA%20Tutorial.ipynb>`_
  - `Stable Diffusion Tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/Stable%20Diffusion%20Tutorial.ipynb>`_
  - `DreamBooth Tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/DreamBooth%20Tutorial.ipynb>`_

Automatic Speech Recognition (ASR) Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Core ASR Techniques and Tools
  - `ASR with NeMo <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_with_NeMo.ipynb>`_
  - `ASR with Subword Tokenization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_with_Subword_Tokenization.ipynb>`_
  - `Offline ASR <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Offline_ASR.ipynb>`_
  - `Online ASR Microphone Cache Aware Streaming <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb>`_
  - `Online ASR Microphone Buffered Streaming <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_ASR_Microphone_Demo_Buffered_Streaming.ipynb>`_
  - `ASR CTC Language Fine-Tuning <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb>`_
  - `Intro to Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Intro_to_Transducers.ipynb>`_
  - `ASR with Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_with_Transducers.ipynb>`_
  - `ASR with Adapters <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/asr_adapters/ASR_with_Adapters.ipynb>`_
  - `Speech Commands <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Speech_Commands.ipynb>`_
  - `Online Offline Microphone Speech Commands <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_Offline_Speech_Commands_Demo.ipynb>`_
  - `Voice Activity Detection <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Voice_Activity_Detection.ipynb>`_
  - `Online Offline Microphone VAD <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_Offline_Microphone_VAD_Demo.ipynb>`_
  - `Speaker Recognition and Verification <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb>`_
  - `Speaker Diarization Inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb>`_
  - `ASR with Speaker Diarization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/ASR_with_SpeakerDiarization.ipynb>`_
  - `Online Noise Augmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_Noise_Augmentation.ipynb>`_
  - `ASR for Telephony Speech <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_for_telephony_speech.ipynb>`_
  - `Streaming inference <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Streaming_ASR.ipynb>`_
  - `Buffered Transducer inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Buffered_Transducer_Inference.ipynb>`_
  - `Buffered Transducer inference with LCS Merge <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Buffered_Transducer_Inference_with_LCS_Merge.ipynb>`_
  - `Offline ASR with VAD for CTC models <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Offline_ASR_with_VAD_for_CTC_models.ipynb>`_
  - `Self-supervised Pre-training for ASR <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Self_Supervised_Pre_Training.ipynb>`_
  - `Multi-lingual ASR <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Multilang_ASR.ipynb>`_
  - `Hybrid ASR-TTS Models <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_TTS_Tutorial.ipynb>`_
  - `ASR Confidence Estimation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_Confidence_Estimation.ipynb>`_
  - `Confidence-based Ensembles <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Confidence_Ensembles.ipynb>`_

Text-to-Speech (TTS) Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Basic and Advanced TTS Topics
  - `NeMo TTS Primer <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/NeMo_TTS_Primer.ipynb>`_
  - `TTS Speech/Text Aligner Inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Aligner_Inference_Examples.ipynb>`_
  - `FastPitch and MixerTTS Model Training <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/FastPitch_MixerTTS_Training.ipynb>`_
  - `FastPitch Finetuning <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/FastPitch_Finetuning.ipynb>`_
  - `FastPitch and HiFiGAN Model Training for German <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/FastPitch_GermanTTS_Training.ipynb>`_
  - `Tacotron2 Model Training <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Tacotron2_Training.ipynb>`_
  - `FastPitch Duration and Pitch Control <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Inference_DurationPitchControl.ipynb>`_
  - `FastPitch Speaker Interpolation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/FastPitch_Speaker_Interpolation.ipynb>`_
  - `TTS Inference and Model Selection <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Inference_ModelSelect.ipynb>`_
  - `TTS Pronunciation_customization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Pronunciation_customization.ipynb>`_

Tools and Utilities
~~~~~~~~~~~~~~~~~~~

* Utility Tools for Speech and Text
  - `NeMo Forced Aligner <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb>`_
  - `Speech Data Explorer <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tools/SDE_HowTo_v2.ipynb>`_
  - `CTC Segmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tools/CTC_Segmentation_Tutorial.ipynb>`_

Text Processing (TN/ITN) Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Text Normalization Techniques
  - `Text Normalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb>`_
  - `Inverse Text Normalization with Thutmose Tagger <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/nlp/ITN_with_Thutmose_Tagger.ipynb>`_
  - `WFST Tutorial <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`_
