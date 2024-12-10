**Introduction**

Welcome to the NeMo SlimPajama Data Pipeline and Pretraining tutorial! This tutorial provides a step-by-step guide to preprocessing the SlimPajama dataset and pretraining a Llama based model using the NeMo 2.0 library.

The tutorial includes two Jupyter notebooks: `data_pipeline.ipynb` and `pretraining.ipynb`. The `data_pipeline.ipynb` notebook provides a data pipeline to preprocess the SlimPajama dataset, including downloading, extracting, concatenating and tokenizing the data. The `pretraining.ipynb` notebook provides a pretraining recipe to train a language model using the preprocessed data.

This repository is designed to be used with the NeMo 2.0 and NeMo-Run.

**Pre-requisites / Requirements**

* Docker
* Python
* NeMo
* NeMo-Run
* Jupyter Lab or Jupyter Notebook

**Getting started**

Assuming you have all the pre-requisites installed, you can get started by following these steps:
1. Clone NeMo to your local machine
2. Navigate to tutorials/llm/llama-3/slimpajama folder inside the repository.
3. Start the Jupyter server by running either `jupyter lab` or `jupyter notebook` based on your preference
4. Follow the directions in data_pipeline.ipynb and pretraining.ipynb notebooks to preprocess the SlimPajama dataset and pretrain a model.

**Note**

* Make sure to replace placeholder paths with the actual paths on your machine.
* The `data_pipeline.ipynb` notebook assumes that the SlimPajama dataset is stored in the `/data/slimpajama` directory.
* The `pretraining.ipynb` notebook assumes that the preprocessed data is stored in the `/data/slimpajama_megatron` directory.