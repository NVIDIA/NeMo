# ChipNeMo - Custom tokenization + Domain Adaptive Pre-training on Llama 2 70b with NeMo Framework

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. Instead of directly deploying off-theshelf
commercial or open-source LLMs, the paper instead adopts the following domain adaptation techniques:
domain-adaptive tokenization, domain adaptive continued pretraining, model alignment with domain-specific instructions, and domain adapted retrieval models. Specifically, LLama 2 foundation models are continually pre-trained with 20B plus tokens on domain-specific chip design data, including code, documents, etc., and then fine-tuned with instruction datasets from design data as well as external sources. Evaluations on the resultant domain-adapted ChipNeMo model demonstrate that
domain-adaptive pretraining of language models, can lead to superior performance in domain related downstream tasks compared to their base Llama 2 counterparts, without degradations in generic capabilities.

Here, we share a tutorial with best practices on custom tokenization + DAPT (domain-adaptive pre-training) for a ChipNeMo-like code generation use case.

* In this tutorial, we will leverage chip domain/hardware datasets from open-source GitHub repositories, wiki URLs, and academic papers.

* `./code/data` should contain curated data from chip domain after processing with NeMo Curator. Playbook for DAPT data curation can be found [here](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/dapt-curation)

* `./code/general_data` should contain open-source general purpose data that llama-2 was trained on. This data will help idenitfy token/vocabulary differences between general purpose and domain-specific datasets. Data can be downloaded from [Wikepedia](https://huggingface.co/datasets/legacy-datasets/wikipedia), [CommonCrawl](https://data.commoncrawl.org/) etc. and curated with [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/single_node_tutorial)

* `./code/custom_tokenization.ipynb` walks through the custom tokenization workflow required for DAPT (Domain Adaptive Pre-training) 