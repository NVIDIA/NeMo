from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from nemo.collections.nlp.models.rag.custom_embedder import NeMoEmbeddings
from nemo.collections.nlp.models.rag.custom_llm import NeMoLLM
import os
import json
from nemo.core.config import hydra_runner


@hydra_runner(config_path="conf", config_name="rag")
def main(cfg) -> None:

    # load LLM
    print("Loading LLM.")
    model_path = cfg.generating.llm.model_path
    Settings.llm = NeMoLLM(model_path = model_path, cfg = cfg)


    # load embedder
    print("Loading embedder.")
    model_path = cfg.indexing.embedder.model_path
    embed_model = NeMoEmbeddings(model_path = model_path, cfg = cfg)
    Settings.embed_model = embed_model


    # load index from disk
    print("Loading index from disk.")
    index_path = cfg.indexing.index_path
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)


    # set query
    print("Setting query.")
    query = cfg.generating.query
    print("Query: ", query)


    # query and print response
    print("Responding to query using neighbors.")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    print(response)


if __name__ == '__main__':
    main()





