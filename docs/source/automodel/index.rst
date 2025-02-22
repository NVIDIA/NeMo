NVIDIA NeMo AutoModel Docs
==========================

NVIDIA NeMo Framework is an end-to-end, cloud-native framework designed to build, customize, and deploy generative AI models anywhere.

NVIDIA NeMo AutoModel supports training and finetuning of models available via 🤗 Transformers' ``AutoModel`` classes,
in particular it supports models available via:

- AutoModelForCausalLM
- AutoModelForImageTextToText
- AutoModelForSpeechSeq2Seq

Covering LLM, VLM and ASR domains.


For distributed processing NeMo AutoModel offers integration with DDP and FSDP2.


For quick guides and tutorials, see the "Getting started" section below.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: starthere
   :titlesonly:

   tutorials/hf_automodel_for_causal_lm_peft
   tutorials/hf_automodel_for_causal_lm_sft



For more information, browse the developer docs for your area of interest in the contents section below or on the left sidebar.

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Code Documentation
   :name: AutoModel Code Documentation

   codedocs/hf_automodel_for_causal_lm
   codedocs/hf_automodel_for_image_text_to_text
   codedocs/hf_automodel_for_speech_seq_to_seq

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Data Documentation
   :name: AutoModel Data Documentation

   codedocs/hf_dataset_data_module

.. toctree::
   :maxdepth: 1
   :caption: AutoModel Callbacks Documentation
   :name: AutoModel Callbacks Documentation

   codedocs/jit_callback
