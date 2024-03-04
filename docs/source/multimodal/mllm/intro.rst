Multimodal Language Models
==========================

The endeavor to extend Language Models (LLMs) into multimodal domains by integrating additional structures like visual encoders has become a focal point of recent research, especially given its potential to significantly lower the cost compared to training multimodal universal models from scratch.

The advent of GPT-4 has spurred a plethora of developments including notable models like LLaVA, Mini-GPT4, and Flamingo. These models, despite minor differences, share similar structural and training strategies.

Supported Models
-----------------
NeMo Multimodal currently supports the following models:

+-----------------------------------+----------+-------------+------+-------------------------+------------------+
| Model                             | Training | Fine-Tuning | PEFT | Evaluation              | Inference        |
+===================================+==========+=============+======+=========================+==================+
| `NeVA (LLaVA) <./neva.html>`_     | Yes      | Yes         | -    | -                       | Yes              |
+-----------------------------------+----------+-------------+------+-------------------------+------------------+

Spotlight Models
-----------------

LLaVA: Visual Instruction Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLaVA :cite:`mm-models-llava` focuses on creating a dataset for visual instruction tuning to enhance LLMs' ability to comprehend diverse instructions and provide detailed responses. NeMo's implementation of LLaVA is called NeVA.

- Model Structure:
    - Visual Encoder: Utilizes CLIP’s ViT-L/14.
    - Text Decoder: Employs LLaMA.
    - Connection: A simple linear mapping layer connects the visual encoder's output to the text decoder's word embedding space (v1.0 version).

- Training:
    1. Cross-modal Pre-training: Utilizes 595k image-text data from CC3M, training only the linear mapping layer while keeping the visual encoder and text decoder frozen.
    2. Instruction Fine-tuning: Custom-built 158k multimodal instruction dataset employed for fine-tuning targeting multimodal chatbot scenarios, with a variant targeting the Science QA dataset.

.. note::
    NeMo Megatron has an Enterprise edition which proffers tools for data preprocessing, hyperparameter tuning, containers, scripts for various clouds, and more. With the Enterprise edition, you also garner deployment tools. Apply for `early access here <https://developer.nvidia.com/nemo-megatron-early-access>`_ .

For more information, see additional sections in the NeMo multimodal language model docs on the left-hand-side menu or in the list below:

.. toctree::
   :maxdepth: 1

   datasets
   configs
   checkpoint
   neva

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-
