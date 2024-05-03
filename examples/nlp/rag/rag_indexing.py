from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from custom_embedder import NeMoEmbeddings
import os


# # download data
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


# load data
print("Loading data.")
documents = SimpleDirectoryReader("/lustre/fsw/coreai_dlalgo_genai/huvu/data/rag_pipeline/sample_data/").load_data()


# set text transformation
print("Setting text transformation.")
Settings.text_splitter = SentenceSplitter()
Settings.chunk_size = 256
Settings.chunk_overlap = 20


# load embedder
print("Loading embedding models.")
model_path = '/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/bert_nemo.nemo'
embed_model = NeMoEmbeddings(model_path = model_path, embed_batch_size = 16)
Settings.embed_model = embed_model


# index data
print("Indexing models.")
## Method 1: default in memory vectorstore
index = VectorStoreIndex.from_documents(documents)
# ## Method 2: Milvus vectorstore
# vector_store = MilvusVectorStore(dim=1536, overwrite=True)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context
# )


# save index data to disk
print("Saving index to disk.")
index_path = "/lustre/fsw/coreai_dlalgo_genai/huvu/data/rag_pipeline/saved_index/sample_index"
index.storage_context.persist(persist_dir=index_path)