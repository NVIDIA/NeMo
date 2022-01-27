.. _prompt_tuning:

Prompt Tuning
====================================

Prompt tuning is a continuous or soft prompt approach to finding the optimal prompt for a specific prompting-based tasks. Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning utilizes continuous prompt tokens that can be optimized via gradient decent. In addition to increased task performance compared to discrete prompting methods, prompt tuning has been shown to yield performance competitive with finetuning all a model’s parameters ([1], [2]) for T5 style models greater than 10B parameters. This is particularly exciting because prompt tuning typically involves tuning parameters amounting to less then 1% of the original model’s size. A model can also be prompt tuned for multiple tasks simultaneously without the risk of over fitting on any one task leading to a degradation in performance on other tasks. With these benefits, prompt tuning can be used as a lighter weight and more flexible alternative to full model finetuning. Prompt tuning can also be used additively with other discrete prompt selection methods.

Our current prompt tuning implementation adapt’s Lester et. al’s paper to prompt tuning for GPT style models. In this implementation, a number of soft tokens specified by the user are prepended to the beginning of the discrete token input embeddings during the forward pass. During training all model parameters are frozen except for those corresponding to the soft tokens. Only the soft prompt parameters are updated via gradient decent in the backward pass.

Example prompt tuning script:
Example prompt tuned inference script: 


Within NeMo we use the entity linking approach described in Liu et. al's NAACL 2021 "`Self-alignment Pre-training for Biomedical Entity Representations <https://arxiv.org/abs/2010.11784v2>`_" :cite:`nlp-entity_linking-liu2021selfalignment`. 
The main idea behind this approach is to reshape an initial concept embedding space such that synonyms of the same concept are 
pulled closer together and unrelated concepts are pushed further apart. The concept embeddings from this reshaped space can then 
be used to build a knowledge base embedding index. 

.. image:: https://github.com/NVIDIA/NeMo/blob/entity-linking-documentation/docs/source/nlp/entity_linking_overview.jpg
  :alt: Entity-Linking-Overview

Our BERT-base + Self Alignment Pretraining implementation allows you to train an entity linking encoder. We also provide example code
on building an index with `Medical UMLS <https://www.nlm.nih.gov/research/umls/index.html>`_ concepts `NeMo/examples/nlp/entity_linking/build_index.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/entity_linking/build_index.py>`__.

Please try the example Entity Linking model in a Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.2/tutorials/nlp/Entity_Linking_Medical.ipynb>`__).

Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

An example script on how to train the model can be found here: `NeMo/examples/nlp/entity_linking <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/entity_linking>`__.


References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: nlp-entity_linking
    :keyprefix: nlp-entity_linking-
