## Supported PEFT methods
NeMo supports the following PFET tuning methods

1. **Attention (Canonical) Adapters**: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
   - Attention Adapters ("Adapters" in the original paper) is one of the first PEFT methods applied to NLP. Adapter 
tuning is more efficient than full fine-tuning because the base model weights are frozen, while only a small number of 
adapter module weights are updated. In this method, two bottleneck modules are inserted into each transformer layer.
Each of them is initialized with near-identity weights to keep training stable. 


2. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685)
   - LoRA makes fine-tuning efficient by representing weight updates with two low rank decomposition matrices. 
The original model weights remain frozen, while the low rank decomposition matrices are updated to adapt to the new data 
, so the number of trainable parameters is kept low. 
In contrast with Attention adapters, the original model weights and adapted weights can be 
combined during inference, avoiding any architectural change or additional latency in the model.
   - The matrix decomposition operation can be applied to any linear layer, but in practice, 
it is only applied to the K, Q, V projection matrices. 
Since NeMo's attention implementation fuses KQV into a single projection, our LoRA implementation learns a 
single Low-Rank projection for KQV in a combined fashion


3. **IA3**: [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](http://arxiv.org/abs/2205.05638)
   - IA3 makes fine-tuning efficient by rescaling activations with learned vectors. The rescaling layers are injected
in the attention (for key and value) and feedforward modules in the base model. 
Similar to LoRA, only the rescaling vectors are updated during fine-tuning to adapt to the new data 
so the number of updated parameters is low. However, since rescaling vectors are much smaller than low rank matrices, 
IA3 cuts down the number of trainable parameters further by an order of magnitude. 
The learning rescaling vectors can also be merged with the base weights,
leading to no architectural change and no additional latency.


4. **P-Tuning**: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
   - P-tuning is an example of the prompt learning family of methods, in which trainable virtual tokens are inserted 
into the model input prompt to induce it to perform a task.
Virtual tokens (also called "continuous" or "soft" tokens) are embeddings that have no concrete mapping to strings 
or characters within the model’s vocabulary. They are simply 1D vectors that match the dimensionality of real tokens
which make up the model's vocabulary.
   - In p-tuning, an LSTM model is used to predict virtual token embeddings. 
We refer to this LSTM model as our `prompt_encoder`. LSTM parameters are randomly initialized at the start of p-tuning. 
All base model parameters are frozen, and only the LSTM weights are updated at each training step. 
LSTM parameters are shared between all tasks that are 
p-tuned at the same time, but the LSTM model outputs unique virtual token embeddings for each task. 
   - You can specify the number of virtual tokens you want to use by setting `total_virtual_tokens` and each virtual 
token embedding is a 1D vector of size `hidden_size`.

