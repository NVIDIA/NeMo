RAG with NeMo
================

Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources. With NeMo, we can employ a text embedder and an LLM trained with NeMo Framework to setup a RAG pipeline.
This document illustrates how NeMo models can be used with LlamaIndex, a popular RAG library, for a retrieval-based text generation application.

## Quick Start

In this example, we set up a pipeline that let us index a document file (e.g., a manual, repository documentation) then ask questions and details from the document.

The only dependency in this example is LlamaIndex, which can be installed with:
```
!pip install llama-index
```

### Indexing data


The first step is creating the index from the corpus document(s). Set the path to the embedder checkpoint, document path, index saving path and relevant arguments, then run the following command. Below we explain in more details the steps run within the script.


```
python examples/nlp/rag/rag_indexing.py \
        trainer.devices=1 \
        trainer.precision='bf16-mixed' \
        indexing.embedder.model_path='/path/to/checkpoints/embedder_model.nemo' \
        indexing.embedder.embed_batch_size=128 \
        indexing.data.data_path='/path/to/data' \
        indexing.data.chunk_size=256 \
        indexing.data.chunk_overlap=10 \
        indexing.index_path='/path/to/index'
```

Followings are the detailed steps ran in the script.

First, the document is read into LlamaIndex's `SimpleDirectoryReader` object.

```
print("Loading documents.")
documents = SimpleDirectoryReader(cfg.indexing.data.data_path).load_data()
```

We then set up how the corpus document(s) will be splitted into smaller chunks, by setting splitter type, chunk size, and chunk overlap values.

```
print("Setting text transformation.")
Settings.text_splitter = SentenceSplitter()
Settings.chunk_size = cfg.indexing.data.chunk_size
Settings.chunk_overlap = cfg.indexing.data.chunk_overlap
```

We then load the embedder NeMo model. Currently, this script only support `.nemo` checkpoints. The wrapper around NeMo LLM to work with LLamaIndex interface is implemented at `nemo/collections/nlp/models/rag/custom_embedder.py`. We can trying different embedding batch size to balance the number of samples embedded at once and embedding speed.

```
print("Loading embedding models.")
model_path = cfg.indexing.embedder.model_path
embed_batch_size = cfg.indexing.embedder.embed_batch_size
embed_model = NeMoEmbeddings(model_path = model_path, cfg = cfg, embed_batch_size = embed_batch_size)
Settings.embed_model = embed_model
```

Next, we will index the corpus document(s), simply by using the LlamaIndex `VectorStoreIndex.from_documents()` method. Under the hood, this method will splitting the document to smaller chunks having with pre-defined chunk size, batch them and feed them to the embedder, then put the output embeddings into the index. In this example, we use the built-in LlamaIndex's in-memory vector store. We can also use external vector stores, such as Milvus, Qdrant. Seeing more at [LlamaIndex Vector Stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/).          


```
print("Indexing data.")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
```

After indexing, we save the index to disk that we can load later to be used with an LLM.

```
print("Saving index to disk.")
index_path = cfg.indexing.index_path
index.storage_context.persist(persist_dir=index_path)
```


###  Generation

After processing and indexing the document, we can have an LLM model to interactive with the doc through RAG, such as asking details within the document. Setting a query to ask and run the following command. Below we explain in more details the steps run within the script.

```
python examples/nlp/rag/rag_eval.py \
        trainer.devices=1 \
        trainer.precision='bf16-mixed' \
        indexing.embedder.model_path='/path/to/checkpoints/embedder_model.nemo' \
        indexing.index_path='/path/to/index' \
        generating.llm.model_path='/path/to/checkpoints/llm_model.nemo' \
        generating.inference.greedy=False \
        generating.inference.temperature=1.0 \
        generating.query='Which art schools did I applied to?'
```

Followings are the detailed steps ran in the script.


First, the LLM is loaded from `generating.llm.model_path`. Currently the script only works with `.nemo` checkpoints. The wrapper around NeMo LLM to work with LLamaIndex interface is implemented at `nemo/collections/nlp/models/rag/custom_llm.py`. 

```
print("Loading LLM.")
model_path = cfg.generating.llm.model_path
Settings.llm = NeMoLLM(model_path = model_path, cfg = cfg)
```

Then we load the index saved on disk in the indexing step. If using Milvus database, it can also be loaded at this step.
```
print("Loading index from disk.")
index_path = cfg.indexing.index_path
storage_context = StorageContext.from_defaults(persist_dir=index_path)
index = load_index_from_storage(storage_context)
```

Finally, we have the LlamaIndex retrieve relevant neighbors and generate answers for the query by the following code piece. With `query` as the input argument, this code piece automatically embed the query with the defined embedder, then retrieve the k relevant neighbors from the index, and add those neighbors to a predefined template along with the query before feeding them to the LLM for generation.
```
print("Responding to query using neighbors.")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query)
print(response)
```

Below is an example of the default template created by LlamaIndex to feed the LLM, which can be modified following LlamaIndex's documentation ;[Prompts RAG](https://docs.llamaindex.ai/en/stable/examples/prompts/prompts_rag/).


```
Context information is below.
---------------------
{context_str 1}
{context_str 2}
...
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer:
```