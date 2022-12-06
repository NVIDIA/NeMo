NeMo RETRO Model
----------------

Retrieval-Enhanced Transformer (RETRO) Model is an auto-regressive language models which can condition 
on document chunks retrieved from a large corpus. For full details of the model, please refer to the Deepmind's 
`RETRO paper <https://arxiv.org/abs/2112.04426>`_.  NeMo RETRO Model is an open source implementation of it, compared 
with Deepmind's proposed implementation, it has the following differences/features: 

1. NeMo RETRO Model is built on top of NeMo Megatron code, which can train large language model efficiently in a cluster environment. 
2. NeMo RETRO Model uses `Faiss <https://github.com/facebookresearch/faiss>`_ as the KNN search library, which can be accelerated by GPU. 
3. NeMo RETRO uses `RoPe relative positional encoding <https://arxiv.org/abs/2104.09864>`_. 
4. NeMo RETRO uses `SentenceTransformers <https://www.sbert.net>`_ as the retriever encoder.
5. NeMo RETRO supports `mu-Transfer <https://openreview.net/pdf?id=Bx6qKuBM2AD>`_, which scale the RETRO model training via Zero-Shot Hyperparameter Transfer.


Quick start
^^^^^^^^^^^
Steps below demonstrate training and evaluate a NeMo RETRO model

Data pre-processing
~~~~~~~~~~~~~~~~~~~

**Step 1: Collect training data**

There are two types of data for the RETRO model: training data (typically has 64 tokens chunks) and retrieval data (typically has 128 tokens chunks).
The training data is used to train the RETRO model, while the retrieval data is used as external data to augment the language model. 
Note that we can use the same data for both training and retrieval, as long as we remove duplicates properly, as explained later.

Both types of raw data are prepared in a loose json format, with one json containing a text sample per line. For example:

.. code-block:: json

    {"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
    {"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}

The name of the text field of the json can be changed by using the ``--json-key`` flag in ``preprocess_data_for_megatron.py``.  The other metadata are optional and are not used in training.

**Step 2: Convert training data into memory map format**

The loose json is then processed into a binary format for training and retrieval. To convert the json into mmap, cached index file. 
Set the ``--dataset-impl`` flag to `retmmap`, which is the memory map format dedicated for RETRO model. 

An example script to prepare data for RETRO training is:

.. code-block:: bash

    python scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=/dataset/pubmed_train.jsonl \
    --json-keys=text \
    --tokenizer-library=megatron \
    --apply-ftfy \
    --dataset-impl=retmmap \
    --merge-file=/dataset/gpt2-merges.txt \
    --vocab-file=/dataset/gpt2-vocab.json \
    --tokenizer-type=GPT2BPETokenizer \
    --output-prefix=/result/pubmed_train \
    --need-pad-id \
    --append-eod \
    --retrieval-db \
    --chunk_size=64 \
    --workers=48

The RETRO model processes chunked documents using 64 tokens as the default chunk size. The RETRO memory map dataset will add padding 
tokens to the end of each document to make it a multiple of 64. The ``--need-pad-id`` argument adds a padding token to the tokenizer
if it doesn't already have one. The ``--append-eod`` argument controls whether to add ``end-of-document`` tokens to the preprocessed 
data, and the ``--retrieval-db`` argument indicates whether to create a retrieval database for the preprocessed data. If ``--retrieval-db``
is used, it will add an additional 64 padding tokens at the end of the document. The ``--chunk_size`` and ``--workers`` arguments 
control the size of the data chunks to be processed and the number of worker processes to use, respectively.


**Step 3: Create Faiss index for retrieval data**

Once we have the memory map retrieval data binary file and index files ready from the previous steps, we can begin to build the Faiss
index that can query the K-nearest neighbors of the chunk IDs given a query embedding vector. Since the retrieval data is typically 
large in size, we break the process down into three sub-steps.

*Step 3.1: Train the Faiss index structure*

In this step, it uses a subset of the retrieval data to train a empty Faiss index. An example script is:

.. code-block:: bash

    python scripts/nlp_language_modeling/build_retrieval_index.py \
    --input_file=/result/pubmed_train_text_document  \
    --tokenizer-library=megatron \
    --tokenizer-type=GPT2BPETokenizer \
    --merge-file=/dataset/gpt2-merges.txt \
    --vocab-file=/dataset/gpt2-vocab.json \
    --percent=1.0 \
    --sentence_transformer_model=all-mpnet-base-v2 \
    --batch_size=1024 \
    --train_index_size=2000000 \
    --workers=2 \
    --devices=0,1,2,3,4,5,6,7 \
    --stage=0 \
    --output_file=/result/pubmed_faiss_learn.index

This will build the empty Faiss index using the ``2000000`` training data in pubmed_train_text_document. 
the ``all-mpnet-base-v2`` sentence transformer model is used to encode the chunk tokens into embedding vector.
The index will be saved in the result directory as ``pubmed_faiss_learn.index``. Here we specify to use 8 GPUs to train
the Faiss index.

*Step 3.2: Add retrieval data into sharding index*

In this step, it adds all the retrieval data into the empty Faiss index created in the previous step.  An example script is:

.. code-block:: bash

    python scripts/nlp_language_modeling/build_retrieval_index.py \
        --input_file=/result/pubmed_train_text_document  \
        --tokenizer-library=megatron \
        --tokenizer-type=GPT2BPETokenizer \
        --merge-file=/dataset/gpt2-merges.txt \
        --vocab-file=/dataset/gpt2-vocab.json \
        --percent=1.0 \
        --sentence_transformer_model=all-mpnet-base-v2 \
        --batch_size=1024 \
        --shard_id=0 \
        --total_shards=10 \
        --workers=2 \
        --devices=0,1,2,3,4,5,6,7 \
        --stage=1 \
        --learned_index=/result/pubmed_faiss_learn.index \
        --output_file=/result/pubmed_faiss_shard0.save

This will break down the retrieval data into ``--total_shards`` shards, and add the data in shard id specified by ``--shard_id``. The 
result will be saved as a file specified by ``--output_file``. In the above example, it will create 10 sharding indexes.

*Step 3.3: Merge the sharding indexes into final Faiss index*

In this step, it merges all the sharding indexes created in the previous step into the final Faiss index.  An example script is:

.. code-block:: bash

    python scripts/nlp_language_modeling/build_retrieval_index.py \
    --stage=2 \
    --devices=0,1,2,3,4,5,6,7 \
    --learned_index=/result/pubmed_faiss_learn.index \
    --shard_index_input=/result/pubmed_faiss_shard \
    --output_file=/result/pubmed_faiss_final.index

**Step 4: Build KNN index**

During training, it is wasteful to run query for KNN chunk IDs for each of the training data point. This can be pre-calculated by 
building the KNN index before training. The KNN index maps the training data chunk id to K-nearest neighbors chunk id in the retrieval 
data. Similarly to building 
