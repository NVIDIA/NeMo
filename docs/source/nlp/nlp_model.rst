.. _nlp_model:

Model NLP
=========

Base class for the rest of the models.

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |  **Description**                                                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_name            | string          | Tokenizer name, will be filled automatically based on model.language_model.pretrained_model_name             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.vocab_file                | string          | Path to tokenizer vocabulary                                                                                 |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_model           | string          | Path to tokenizer model (only for sentencepiece tokenizer)                                                   |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.pretrained_model_name| string          | Pre-trained language model name, for example: `bert-base-cased` or `bert-base-uncased`                       |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.lm_checkpoint        | string          | Path to the pre-trained language model checkpoint                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config_file          | string          | Path to the pre-trained language model config file                                                           |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config               | dictionary      | Config of the pre-trained language model                                                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

Important Parameters
^^^^^^^^^^^^^^^^^^^^

Below is the list of parameters could help improve your MLP model:

- language model (`model.language_model.pretrained_model_name`)
    - pre-trained language model name, such as:
    - `megatron-bert-345m-uncased`, `megatron-bert-345m-cased`, `biomegatron-bert-345m-uncased`, `biomegatron-bert-345m-cased`, `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`, `bert-large-cased`
    - `distilbert-base-uncased`, `distilbert-base-cased`,
    - `roberta-base`, `roberta-large`, `distilroberta-base`
    - `albert-base-v1`, `albert-large-v1`, `albert-xlarge-v1`, `albert-xxlarge-v1`, `albert-base-v2`, `albert-large-v2`, `albert-xlarge-v2`, `albert-xxlarge-v2`

- classification head parameters:
    - the number of layers in the classification head (`model.head.num_fc_layers`)
    - dropout value between layers (`model.head.fc_dropout`)

- optimizer (`model.optim.name`, for example, `adam`)
- learning rate (`model.optim.lr`, for example, `5e-5`)

