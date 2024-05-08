RAG with NeMo
================

After training embedder and LLM with NeMo. We can put them to use in a RAG pipeline. 
Below is an example of using NeMo embedder and LLM with LLaMa Index, a popular library for RAG application.
Assume that we have the SBERT Embedding model, at:
For LLM, we can train use model at:

Currently, work with .nemo checkpoint.


Quick Start
************
The following instructions demonstrate how to preprocess the data as well as train and evaluate a RETRO model.

Indexing data
-------------------

Process data. Load embedder. Extract embeddings and save to index. Save index to disk. The input is a text file.

```
python examples/nlp/rag/rag_indexing.py




```

This process include:
Process, load data:

```
documents = SimpleDirectoryReader("/lustre/fsw/coreai_dlalgo_genai/huvu/data/rag_pipeline/sample_data/").load_data()
```

Chunksize will be set as:
```
print("Setting text transformation.")
Settings.text_splitter = SentenceSplitter()
Settings.chunk_size = 256
Settings.chunk_overlap = 10
```

Load embedder. .nemo checkpoint. Needs a wrapper around NeMo model to work with LLamaIndex, at `custom_embedder`.
```
model_path = '/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/bert_nemo.nemo'
embed_model = NeMoEmbeddings(model_path = model_path, embed_batch_size = 128)
Settings.embed_model = embed_model
```

We then index data. This process chunk the documents into smaller chunks and feed them to emebedder, then put the index. 
In this example, LLamaIndex's built-in datastore, for more, we can use open source vector store such as Milvus.
```
# index data
print("Indexing data.")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
```

Save index to disk.
```
index_path = "/lustre/fsw/coreai_dlalgo_genai/huvu/data/rag_pipeline/saved_index/sample_index"
index.storage_context.persist(persist_dir=index_path)
```


Evaluating data
-----------------------

With the document read and embedded, indexed, we ask question about that doc.

We will load LLM, load embedder, load index.

We run this command.


```
python examples/nlp/rag/rag_eval.py


```

Load LLM model. Works with .nemo only. Needs a wrapper around NeMo model to work with LLamaIndex, at `custom_llm`.





