# Parameter-Efficient Fine-Tuning (PEFT) in NeMo

PEFT is a popular technique to efficiently finetune large language models for use in various downstream tasks.
The idea is to freeze the base model weights during finetuning, but inject a few adapters modules to the model 
which result in a small number (< 1%) of (additional) trainable weights.
With carefully chosen adapter modules and injection points, PEFT achieves comparable performance to full finetuning 
at a fraction of the computational and storage costs.

Learn more about PEFT in NeMo with the 
[Quick Start Guide](quick_start.md).
Or, if you wish to follow a practical example, take a look at the
[Step-by-step Guide](lora_tutorial.md).

## Supported PEFT methods
NeMo supports the following PFET tuning methods
TODO add a few sentences to describe each paper

- **Attention (Canonical) Adapter**: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685)
- **IA3**: [Infused Adapter by Inhibiting and Amplifying Inner Activations](http://arxiv.org/abs/2205.05638)
- **P-Tuning**: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)


## Supported Models
- GPT: `MegatronGPTSFTModel`
- T5: `MegatronT5SFTModel`








