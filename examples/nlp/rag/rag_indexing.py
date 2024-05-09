from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from nemo.collections.nlp.models.rag.custom_embedder import NeMoEmbeddings
import os
import pandas as pd
from nemo.core.config import hydra_runner

@hydra_runner(config_path="conf", config_name="rag")
def main(cfg) -> None:

    # load data
    print("Loading documents.")
    documents = SimpleDirectoryReader(cfg.indexing.data.data_path).load_data()


    # set text transformation
    print("Setting text transformation.")
    Settings.text_splitter = SentenceSplitter()
    Settings.chunk_size = cfg.indexing.data.chunk_size
    Settings.chunk_overlap = cfg.indexing.data.chunk_overlap


    # load embedder
    print("Loading embedding models.")
    model_path = cfg.indexing.embedder.model_path
    embed_batch_size = cfg.indexing.embedder.embed_batch_size
    embed_model = NeMoEmbeddings(model_path = model_path, cfg = cfg, embed_batch_size = embed_batch_size)
    Settings.embed_model = embed_model


    # index data
    print("Indexing data.")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)


    # save index data to disk
    print("Saving index to disk.")
    index_path = cfg.indexing.index_path
    index.storage_context.persist(persist_dir=index_path)


if __name__ == '__main__':
    main()


