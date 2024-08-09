
# Performance Benchmarks

## Large Language Models**

### Pretraining

- The results in the table below show pre-training performance (using NeMo Framework 24.07) of various models on DGXH100, with FP8 precision.
- Please refer to `MLCommons Training results <https://mlcommons.org/benchmarks/training/>`_ for performance of GPT3-175B pre-training on large scale H100 systems.

| Model         | #-GPUs | GBS  | MBS | Sequence Length| TP | PP | CP | Tokens / sec / GPU | Model TFLOP / sec / GPU | ***Est. time to train in days (10T tokens, 1K GPUs)*** |
| -----         | ------ | ---  | --- | ---------------| -- | -- | -- | ------------------ | ----------------------- | ------------------------------------------------------ |
| GPT3-5B       | 64     | 2048 | 4   | 2048           | 1  | 1  | 1  | 22521              | 736                     | ***5***                                                |
| GPT3-20B      | 64     | 256  | 2   | 2048           | 2  | 1  | 1  | 5851               | 750                     | ***19***                                               |
| LLAMA2-7B     | 8      | 128  | 1   | 4096           | 1  | 1  | 1  | 16847              | 776                     | ***7***                                                | 
| LLAMA2-13B    | 16     | 128  | 1   | 4096           | 1  | 4  | 1  | 8646               | 754                     | ***13***                                               |
| LLAMA2-70B    | 64     | 128  | 1   | 4096           | 4  | 4  | 1  | 1707               | 759                     | ***66***                                               |
| Nemotron-8B   | 64     | 256  | 4   | 4096           | 2  | 1  | 1  | 12701              | 653                     | ***9***                                                |
| Nemotron-22B  | 64     | 256  | 2   | 4096           | 2  | 4  | 1  | 4256               | 554                     | ***27***                                               |
| Nemotron-340B | 128    | 32   | 1   | 4096           | 8  | 8  | 1  | 322                | 678                     | ***351***                                              |
| LLAMA3-8B     | 8      | 128  | 1   | 8192           | 1  | 1  | 2  | 12036              | 697                     | ***9***                                                |
| LLAMA3-70B    | 64     | 128  | 1   | 8192           | 4  | 4  | 2  | 1533               | 738                     | ***74***                                               |

### Finetuning

- The following table provides performance benchmarking of LLAMA2 models with SFT (supervised fine-tuning), and LoRA (Low-rank adaptors) on DGXH100, with FP8.
- For fine-tuning, we use `SQuAD-v1.1 <https://rajpurkar.github.io/SQuAD-explorer/>`__ dataset, and the inputs are packed to 4096 tokens.


| Model      | Mode     | #-GPUs | GBS | MBS | Sequence Length | TP | PP | Tokens / sec / GPU | Model TFLOP / sec / GPU |  ***Est. time to finetune in mins (10M tokens)***  |
| -----      | ----     | ---    | --- | --- | --------------- | -- | -- | ------------------ | ----------------------- | -------------------------------------------------- |
| LLAMA2-7B  | SFT      | 8      | 32  | 1   | 4096            | 1  | 1  | 17120              | 682                     | ***1.2***                                          |
| LLAMA2-13B | SFT      | 8      | 32  | 1   | 4096            | 1  | 4  | 9256               | 716                     | ***2.2***                                          |
| LLAMA2-70B | SFT      | 16     | 32  | 1   | 4096            | 4  | 4  | 1833               | 756                     | ***5.7***                                          |
| LLAMA2-7B  | LoRA     | 8      | 32  | 1   | 4096            | 1  | 1  | 25206              | 673                     | ***0.8***                                          |
| LLAMA2-13B | LoRA     | 8      | 32  | 1   | 4096            | 1  | 1  | 13653              | 707                     | ***1.5***                                          |
| LLAMA2-70B | LoRA     | 8      | 32  | 1   | 4096            | 2  | 4  | 2470               | 681                     | ***8.4***                                          |
                                                      |
