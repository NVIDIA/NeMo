Flash Attention
---------------

Overview
^^^^^^^^

Flash Attention :cite:`nlp-megatron-dao2022flashattention` is a method designed to enhance the efficiency of Transformer models, which are widely utilized in applications such as natural language processing. Traditional Transformers are slow and consume a lot of memory, especially with long sequences, due to the quadratic time and memory complexity of self-attention. FlashAttention, an IO-aware exact attention algorithm that leverages tiling to minimize the number of memory reads/writes between the GPU's high bandwidth memory (HBM) and on-chip SRAM. This approach is designed to be more efficient in terms of IO complexity compared to standard attention mechanisms.

Turn on and off Flash Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NeMo framework, Flash Attention is supported through the Transformer Engine with the inclusion of Flash Attention 2. By default, Flash Attention is enabled, but the Transformer Engine may switch to a different kernel if the tensor dimensions are not optimal for Flash Attention. Users can completely disable Flash Attention by setting the environment variable `NVTE_FLASH_ATTN=0`.

For more details on the supported Dot Attention backend, please refer to the Transformer Engine source code available at `Transformer Engine's Attention Mechanism <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.

.. bibliography:: nlp_all.bib
