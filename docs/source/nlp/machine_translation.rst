.. _machine_translation:

Machine Translation Models
=========================
Machine translation is the task of translating text from one language to another. For example, from English to Spanish.
Models are based on the Transformer sequence-to-sequence architecture :cite:`nlp-machine_translation-vaswani2017attention`.

An example script on how to train the model can be found here: `NeMo/examples/nlp/machine_translation/enc_dec_nmt.py <https://github.com/NVIDIA/NeMo/blob/r1.0.0rc1/examples/nlp/machine_translation/enc_dec_nmt.py>`__.
The default configuration file for the model can be found at: `NeMo/examples/nlp/machine_translation/conf/aayn_base.yaml <https://github.com/NVIDIA/NeMo/blob/r1.0.0rc1/examples/nlp/machine_translation/conf/aayn_base.yaml>`__.

Quick Start
-----------

.. code-block:: python

    from nemo.collections.nlp.models import MTEncDecModel

    # To get the list of pre-trained models
    MTEncDecModel.list_available_models()

    # Download and load the a pre-trained to translate from English to Spanish
    model = MTEncDecModel.from_pretrained("nmt_en_es_transformer12x2")

    # Translate a sentence or list of sentences
    translations = model.translate(["Hello!"], source_lang="en", target_lang="es")

Available Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - English -> German
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer12x2
   * - German -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer12x2
   * - English -> Spanish
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer12x2
   * - Spanish -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer12x2
   * - English -> French
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer12x2
   * - French -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer12x2
   * - English -> Russian
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer6x6
   * - Russian -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer6x6
   * - English -> Chinese
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer6x6
   * - Chinese -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer6x6

Data Format
-----------
Supervised machine translation models require parallel corpora which comprise many examples of sentences in a source language and their corresponding translation in a target language.
We use parallel data formatted as seperate text files for source and target language where sentences in corresponding files are aligned like in the table below.

.. list-table:: *Parallel Coprus*
   :widths: 10 10
   :header-rows: 1

   * - train.english.txt
     - train.spanish.txt
   * - Hello .
     - Hola .
   * - Thank you .
     - Gracias .
   * - You can now translate from English to Spanish in NeMo .
     - Ahora puedes traducir del inglés al español en NeMo .

It is common practice to apply data cleaning, normalization and tokenization to the data prior to training a translation model and NeMo expects already cleaned, normalized and tokenized data.
The only data pre-processing NeMo does is subword tokenization with BPE :cite:`nlp-machine_translation-sennrich2015neural`.

Data Cleaning, Normalization & Tokenization
-----------

We recommend applying the following steps to clean, normalize and tokenize your data. All pre-trained models released apply these data pre-processing steps.

1. Language ID filtering - This step filters out examples from your training dataset that aren't in the correct language.
For example, many datasets contain examples where source and target sentences are in the same language . You can use a pre-trained language ID classifier from `fastText <https://fasttext.cc/docs/en/language-identification.html>`__.
You can then run our script using the `lid.176.bin` model downloaded from the fastText website.

.. code ::

    python NeMo/scripts/neural_machine_translation/filter_by_language.py \
      --input-src train.en \
      --input-tgt train.es \
      --output-src train_lang_filtered.en \
      --output-tgt train_lang_filtered.es \
      --source-lang en \
      --target-lang es \
      --removed-src train_noise.en \
      --removed-tgt train_noise.de \
      --fasttext-model lid.176.bin

2. Length filtering - We filter out sentences from the data that are below a minimum length (1) or exceed a maximum length (250).
We also filter out sentences where the ratio between source and target lengths exceeds 1.3 except for English <-> Chinese models.
`Moses <https://github.com/moses-smt/mosesdecoder>`__ is a statistical machine translation toolkit that contains many useful pre-processing scripts.

.. code ::

    perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.3 train en es train.filter 1 250

3. Data cleaning - While language ID filtering can sometimes help with filtering out noisy sentences that contain too many puncutations, it does not help in cases where the translations are potentially incorrect, disfluent or incomplete.
We use `bicleaner <https://github.com/bitextor/bicleaner>`__ a tool to identify such sentences. It trains a classifier based on many features included pre-trained language model fluency, word alignment scores from a word-alignment model like `Giza++ <https://github.com/moses-smt/giza-pp>`__ etc.
We use their available pre-trained models wherever possible and train models ourselves using their framework for remaining languages. The following script will apply a pre-trained bicleaner model to the data and pick sentences that are clean with probability > 0.5.

.. code ::

    awk '{print "-\t-"}' train.en \
    | paste -d "\t" - train.filter.en train.filter.es \
    | bicleaner-classify - - </path/to/bicleaner.yaml> > train.en-es.bicleaner.score

4. Data deduplication - We use `bifixer <https://github.com/bitextor/bifixer>`__ (which uses xxHash) to hash the source and target sentences based on which we remove duplicate entries from the file.
You may want to do something similar to remove training examples that are in the test dataset.

.. code ::

    cat train.en-es.bicleaner.score \
      | parallel -j 25 --pipe -k -l 30000 python bifixer.py --ignore-segmentation -q - - en es \
      > train.en-es.bifixer.score
    
    awk -F awk -F "\t" '!seen[$6]++' train.en-es.bifixer.score > train.en-es.bifixer.dedup.score


Filter out data that bifixer assigns probability < 0.5 to.

.. code ::

    awk -F "\t" '{ if ($5>0.5) {print $3}}' train.en-es.bifixer.dedup.score > train.cleaned.en
    awk -F "\t" '{ if ($5>0.5) {print $4}}' train.en-es.bifixer.dedup.score > train.cleaned.es


5. Punctuation Normalization - Punctuation, especially things like quotes can be written in different ways and its often useful to normalize the way they appear in text. We use the moses punctuation normalizer on all languages except Chinese.

.. code ::

    perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l es < train.cleaned.es > train.normalized.es
    perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < train.cleaned.en > train.normalized.en

Example:

.. code ::

    Before - Aquí se encuentran joyerías como Tiffany`s entre negocios tradicionales suizos como la confitería Sprüngli.
    After  - Aquí se encuentran joyerías como Tiffany's entre negocios tradicionales suizos como la confitería Sprüngli.

6. Tokenization and word segmentation for Chinese

Naturally written text often contains punctuation markers like commas, full-stops and apostrophes that are attached to words. Tokenization by just splitting a string on spaces will result in separate token IDs for very similar items like "NeMo" and "NeMo.".
Tokenization splits punctuation from the word to create two separate tokens. In the previous example "NeMo." becomes "NeMo ." which when split by space results in two tokens and adressess the earlier problem. Example:

.. code ::

    Before - Especialmente porque se enfrentará "a Mathieu (Debuchy), Yohan (Cabaye) y Adil (Rami) ", recuerda.
    After  - Especialmente porque se enfrentará " a Mathieu ( Debuchy ) , Yohan ( Cabaye ) y Adil ( Rami ) " , recuerda .

We use the Moses tokenizer for all languages except Chinese.

.. code ::

    perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l es -no-escape < train.normalized.es > train.tokenized.es
    perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape < train.normalized.en > train.tokenized.en

For languages like Chinese where there is no explicit marker like spaces that separate words, we use `Jieba <https://github.com/fxsjy/jieba>`__ to segment a string into words that are space separated. Example:

.. code ::

    Before - 同时，卫生局认为有必要接种的其他人员，包括公共部门，卫生局将主动联络有关机构取得名单后由卫生中心安排接种。
    After  - 同时 ， 卫生局 认为 有 必要 接种 的 其他 人员 ， 包括 公共部门 ， 卫生局 将 主动 联络 有关 机构 取得 名单 后 由 卫生 中心 安排 接种 。


Training a BPE Tokenization
------------------
Byte-pair encoding (BPE) :cite:`nlp-machine_translation-sennrich2015neural` is a sub-word tokenization algorithm that is commonly used to reduce the large vocabulary size of datasets by splitting words into frequently occuring sub-words.
Currently, Machine Translation only supports the `YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__ BPE tokenizer. One can set the tokenization configuration as follows:

+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| **Parameter**                                               | **Data Type**   |   **Default**  | **Description**                                                                                |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{encoder_tokenizer,decoder_tokenizer}.tokenizer_name  | str             | yttm           | BPE library name. Only supports yttm for now.                                                  |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{encoder_tokenizer,decoder_tokenizer}.tokenizer_model | str             | null           | Path to an existing YTTM BPE model. If null, will train one from scratch on the provided data. |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{encoder_tokenizer,decoder_tokenizer}.vocab_size      | int             | null           | Desired vocabulary size after BPE tokenization                                                 |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{encoder_tokenizer,decoder_tokenizer}.bpe_dropout     | float           | null           | BPE dropout probability. :cite:`nlp-machine_translation-provilkov2019bpe`                      |   
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{encoder_tokenizer,decoder_tokenizer}.vocab_file      | str             | null           | Path to pre-computed vocab file if exists                                                      |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.shared_tokenizer                                      | bool            | True           | Whether to share the tokenizer between the encoder and decoder                                 |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+


Applying BPE Tokenization, batching, bucketing and padding
------------------
Given BPE tokenizers, and a cleaned parallel corpus, the following steps are applied to create a a `TranslationDataset <https://github.com/NVIDIA/NeMo/blob/r1.0.0rc1/nemo/collections/nlp/data/machine_translation/machine_translation_dataset.py#L64>`__ object.

1. Text to IDs - This performs subword tokenization with the BPE model on an input string and maps it to a sequence of tokens for the source and target text.

2. Bucketing - Sentences vary in length and when creating minibatches, we'd like sentences in them have roughly the same length to minimize the number of <pad> tokens, to maximize computational efficiency. This step groups sentences on roughly the same length into buckets.

3. Batching and padding - Creates minibatches of with a maximum number of tokens specified by `model.{train_ds,validation_ds,test_ds}.tokens_in_batch` from buckets and pads the sequences to pack them so they can be packed into a tensor.

Datasets can be configured as follows

+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| **Parameter**                                               | **Data Type**   |   **Default**  | **Description**                                                                                |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.src_file_name        | str             | null           | Path to the source language file                                                               |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.tgt_file_name        | str             | null           | Path to the target language file                                                               |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.tokens_in_batch      | int             | 512            | Maximum number of tokens per minibatch                                                         |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.clean                | bool            | true           | Whether to clean the dataset by discarding examples that are greater than max_seq_length       |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.max_seq_length       | int             | 512            | Path to pre-computed vocab file if exists                                                      |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.cache_ids            | bool            | true           | Whether to cache IDs to avoid re-tokenizing data                                               |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.cache_data_per_node  | bool            | false          | Whether to cache IDs in each of the nodes in multi-node training                               |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.use_cache            | bool            | true           | Whether to use the cache if available                                                          |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.shuffle              | bool            | true           | Whether to shuffle minibatches in the PyTorch DataLoader                                       |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.num_samples          | int             | -1             | Number of samples to use. -1 for the entire dataset                                            |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.drop_last            | bool            | false          | Drop last minibatch if it is not of equal size to the others                                   |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.pin_memory           | bool            | false          | Whether to pin memory in the PyTorch DataLoader                                                |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+
| model.{train_ds,validation_ds,test_ds}.num_workers          | int             | 8              | Number of workers for the PyTorch DataLoader                                                   |
+-------------------------------------------------------------+-----------------+----------------+------------------------------------------------------------------------------------------------+


Tarred Datasets for Large Corpora
------------------

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: nlp-machine_translation
    :keyprefix: nlp-machine_translation-
