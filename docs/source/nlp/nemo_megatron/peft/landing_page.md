# Parameter-Efficient Fine-Tuning (PEFT) in NeMo

PEFT is a popular technique used to efficiently finetune large language models for use in various downstream tasks.
When finetuning with PEFT, the base model weights are frozen, and a few trainable adapter modules are injected 
into the model, resulting in a very small number (<< 1%) of trainble weights.
With carefully chosen adapter modules and injection points, PEFT achieves comparable performance to full finetuning 
at a fraction of the computational and storage costs.

NeMo supports four PEFT methods which can be used with two transformer-based networks.

|     | GPT | T5  |
|-----|-----|-----|
| Attn Adapter    | ✅   | ✅    |
| LoRA    | ✅   |  ✅   |
| IA3    |  ✅   |  ✅   |
| P-Tuning    | ✅    | ✅    |

Learn more about PEFT in NeMo with the 
[Quick Start Guide](quick_start.md) which provides an overview on how PEFT works in NeMo.
Read about the supported PEFT methods [here](supported_methods.md).
For a practical example, take a look at the
[Step-by-step Guide](lora_tutorial.md).








