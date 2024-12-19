# Training Code for DAPT (Domain Adaptive Pre-Training)

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. Instead of directly deploying off-theshelf commercial or open-source LLMs, the paper instead adopts the following domain adaptation techniques: domain-adaptive tokenization, domain adaptive continued pretraining, model alignment with domain-specific instructions, and domain adapted retrieval models. Specifically, LLama 2 foundation models are continually pre-trained with 20B plus tokens on domain-specific chip design data, including code, documents, etc., and then fine-tuned with instruction datasets from design data as well as external sources. Evaluations on the resultant domain-adapted ChipNeMo model demonstrate that domain-adaptive pretraining of language models, can lead to superior performance in domain related downstream tasks compared to their base LLaMA2 counterparts, without degradations in generic capabilities.

Here, we share a tutorial with best practices on training for DAPT (domain-adaptive pre-training).


## Prepare Real Dataset
You can use dummy data for testing. 
Please see [create dummy data](./Step0_Dummy_Data.ipynb)

### Domain Specific Data
Please see [NeMo-Curator DAPT](https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/dapt-curation)

### General Purpose Data: Wiki

```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

pip install wikiextractor
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 --json
find text -name 'wiki_*' -exec cat {} \; > train_data.jsonl
```

### Model Alignment: Oassat & HelpSteer data 

```bash
python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=/work/Data/oasst
python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_helpsteer_data.py --output_directory=/work/Data/helpsteer

cat /work/Data/oasst/train.jsonl /work/Data/helpsteer/train.jsonl | awk '{for(i=1;i<=4;i++) print}' > /work/Data/merge_steerlm_train.jsonl
cat /work/Data/oasst/val.jsonl /work/Data/helpsteer/val.jsonl > /work/Data/merge_steerlm_val.jsonl
rm -rf /work/Data/oasst
rm -rf /work/Data/helpsteer

python /opt/NeMo-Aligner/examples/nlp/data/steerlm/process_to_regression_format.py \
   --input-file=/work/Data/merge_steerlm_train.jsonl \
   --output-file=/work/Data/merge_steerlm_train_reg.jsonl

python /opt/NeMo-Aligner/examples/nlp/data/steerlm/process_to_regression_format.py \
   --input-file=/work/Data/merge_steerlm_val.jsonl \
   --output-file=/work/Data/merge_steerlm_val_reg.jsonl
```