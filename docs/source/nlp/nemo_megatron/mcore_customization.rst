Megatron Core Customization
---------------------------

Megatron offers a range of functionalities, one of the most notable being the ability for users to train GPT models on an epic scale. Users can use megatron.core.models.gpt.GPTModel (mcore GPTModel) to initialize the model, and then pretrain/load weights into the model. Mcore GPTModel adopts the typical GPT structure, beginning with embedding layer, positional encoding, followed by a series of transformer layers and finally output layer. 

In the rapidly advancing world of LLM, it is increasingly important to experiment with various configurations of the transformer block within each transformer layer. Some of these configurations involve the use of different module classes. While it is possible to achieve this with “if else” statements in mcore, doing so makes mcore less readable and less maintainable in the long term. Mcore spec intends to solve this challenge by allowing users to specify a customization of the transformer block in each layer, without modifying code in mcore. 
We will dive more into the details of mcore spec in the first section of this blog. Then, we will demonstrate the usefulness of mcore spec using Falcon as an example.

What is Mcore Spec
^^^^^^^^^^^^^^^^^^

Submodules
""""""""""

ModuleSpec
""""""""""

Build Module
""""""""""""

Customization Examples
^^^^^^^^^^^^^^^^^^^^^^

Customize model initialization
""""""""""""""""""""""""""""""

Customize model forward
"""""""""""""""""""""""
