Memory Optimizations
====================

Parallelism
-----------
Refer to :doc:`Parallelism <./parallelisms>`.

Flash Attention
---------------

Overview
^^^^^^^^

Flash attention is an algorithm designed to improve the efficiency of the attention mechanism in transformer models such as GPT and BERT. The attention mechanism has quadratic time and memory complexity in sequence length and can present significant runtime and memory challenges when training with long sequences. The flash attention algorithm, compared to standard attention, applies two techniques to lower the memory requirement and improve the compute efficiency. (i) Tiling: instead of working on the entire query, key, value tensors, it makes several passes at those input tensors and calculates the softmax one tile at a time. (ii) Recomputation: instead of storing the softmax matrix (qudratic to sequence length), it stores the softmax normalization factors (linear to sequence length) and recomputes attention scores using these factors in the backward pass. Despite an increased number of FLOPs due to recomputation, flash attention improves the runtime efficiency by minimizing the number of reads and writes between HBM and shared memory on the GPU. It also reduces the memory footprint from quadratic to linear to sequence length, greatly expanding the range of sequence length allowed for transformer models.

The flash attention algorithm was first propsed `here <https://arxiv.org/pdf/2205.14135>`_, and two of its implementations are `flash-attention <https://github.com/Dao-AILab/flash-attention>`_ by Tri Dao *et al*, and `fused flash attention <https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/developer-guide/index.html#flash-fused-multi-head-att-fprop>`_ by NVIDIA cuDNN.

Turn Flash Attention On and Off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NeMo Framework, flash attention is supported through `Transformer Engine <https://github.com/NVIDIA/TransformerEngine/tree/main>`_ with both of the above implementations. Transformer Engine selects the appropriate implementation based on the input information (sequence length, number of heads, head dimension, etc), but when both implementations are applicable, Transformer Engine prefers cuDNN flash attention on Hopper+ architectures, and Tri Dao flash attention on Ampere-based architectures. To disable Tri Dao flash attention, users can set the environment variable ``NVTE_FLASH_ATTN=0``, and to disable cuDNN flash attention, users can set ``NVTE_FUSED_ATTN=0``.

For more details on the Dot Product Attention backends supported in Transformer Engine, please refer to the source code at `Transformer Engine's Attention Mechanism <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.

Activation Recomputation
------------------------

Overview
^^^^^^^^

Full Activation Recomputation
"""""""""""""""""""""""""""""
This method recalculates all the intermediate activations during the backward pass of a model's training, instead of storing them during the forward pass. This technique maximizes memory efficiency at the cost of computational overhead, as each activation is recomputed when needed.

Partial Activation Recomputation
""""""""""""""""""""""""""""""""
This method recomputes only a subset of layers during the backward phase. It is a trade-off between the full recomputation and no recomputation, balancing memory savings with computational efficiency.

Selective Activation Recomputation
""""""""""""""""""""""""""""""""""
This method reduces memory footprint of activations significantly via smart activation checkpointing. This approach involves selectively storing only crucial activations and recomputing the others as needed. It is particularly useful in large models to minimize memory usage while controlling the computational cost.

Refer to "Reducing Activation Recomputation in Large Transformer Models" for more details: https://arxiv.org/abs/2205.05198
