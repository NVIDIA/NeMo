# Parameter-Efficient Fine-Tuning (PEFT) in NeMo

PEFT is a popular technique used to efficiently finetune large language models for use in various downstream tasks.
When finetuning with PEFT, the base model weights are frozen, and a few trainable adapter modules are injected 
into the model, resulting in a very small number (< 1%) of trainble weights.
With carefully chosen adapter modules and injection points, PEFT achieves comparable performance to full finetuning 
at a fraction of the computational and storage costs.

Learn more about PEFT in NeMo with the 
[Quick Start Guide](quick_start.md) which provides an overview on how PEFT works in NeMo.
For a practical example, take a look at the
[Step-by-step Guide](lora_tutorial.md).

## Supported PEFT methods
NeMo supports the following PFET tuning methods

1. **Attention (Canonical) Adapter**: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
   - todo


2. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685)
   - LoRA represents weight matrices with two low rank decomposition matrices. The low rank decomposition updates the 
original, frozen weight matrix in order to keep the number of parameters low.
   - The decomposition can be applied to any linear layer, but in practice, it is only applied to the K, Q, V projection 
matrices. Since NeMo's attention implementation fuses KQV into a single projection, our LoRA implementation learns a 
single Low-Rank projection for KQV in a combined fashion


3. **IA3**: [Infused Adapter by Inhibiting and Amplifying Inner Activations](http://arxiv.org/abs/2205.05638)
   - todo


4. **P-Tuning**: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
   - todo


## Supported Models
- GPT: `MegatronGPTSFTModel`
- T5: `MegatronT5SFTModel`








