from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from custom_embedder import NeMoEmbeddings
from custom_llm import NeMoLLM
import os


# load LLM
print("Loading LLM.")
model_path = "/lustre/fsw/coreai_dlalgo_ci/nemo_infer_container/LLAMA2-7B-base/LLAMA2-7B-base-1.nemo"
Settings.llm = NeMoLLM(model_path)


# load embedder
print("Loading embedder.")
model_path = '/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/bert_nemo.nemo'
embed_model = NeMoBertEmbeddings(model_path)
Settings.embed_model = embed_model


# load index from disk
print("Loading index from disk.")
index_path = "/lustre/fsw/coreai_dlalgo_genai/huvu/data/rag_pipeline/saved_index/sample_index"
storage_context = StorageContext.from_defaults(persist_dir=index_path)
index = load_index_from_storage(storage_context)


# set query
print("Setting query.")
# query = "When did I submitt the camera-ready copy of ANSI Common Lisp to the publishers?"
# query = "Who was in charge of marketing at a Boston investment bank?"
# query = "What was the worst thing about leaving YC?"
# query = "What was the year and month did I run into professor Cheatham and he asked if I was far enough along to graduate that June."
# query = "What did Cornell do to Robert Morris after he wrote the internet worm of 1988?"
query = "Which art schools did I applied to?"
print("Query: ", query)


# query and print response
print("Responding to query using neighbors.")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query)
print(response)
