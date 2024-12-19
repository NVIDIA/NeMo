from llama_index.core import Settings, StorageContext, load_index_from_storage
from nemo.collections.nlp.models.rag.custom_bert_embedder import NeMoBertEmbeddings
from nemo.collections.nlp.models.rag.custom_gpt_llm import NeMoGPTLLM
import os
import random
import json
import tqdm
import logging
from nemo.core.config import hydra_runner

# Constants
QUERY_PROMPT_TEMPLATE = """You will be provided with a document or a passage. Your task is to generate a single, highly relevant and natural language query that aligns perfectly with the content of the document. The query should:
    1. Reflect the main idea or a key detail from the document.
    2. Be concise, specific, and written in natural language.
    3. Be something a user might naturally ask to retrieve this document.
 
 ## Given Document:
 {document}
 
 ## Predict Query:
 """
 
FILTER_PROMPT_TEMPLATTE = """You will be provided with a document and a query.
Your task is to evaluate whether the content of the document is relevant to answering the query.
Return True if the document contains information directly related to the query, and False if it does not.

## Given Document:
{document}

## Given Query:
{query}

## Answer(True or False):
"""

# Helper functions
def get_random_document_from_index(index):
    # Access the nodes_dict from index_struct
    nodes_dict = index.index_struct.nodes_dict

    # Check if nodes_dict is empty
    if not nodes_dict:
        return "No documents in the index."

    # Extract node IDs
    node_ids = list(nodes_dict.keys())

    # Select a random node ID
    random_node_id = random.choice(node_ids)

    # Retrieve the corresponding node from the docstore
    random_node = index.docstore.get_node(random_node_id)
    return random_node.text

def get_random_documents_from_index(index, count):
    # nodes = list(index.index_struct.values())
    # Access the nodes_dict from index_struct
    nodes_dict = index.index_struct.nodes_dict

    # Check if nodes_dict is empty
    if not nodes_dict:
        return "No documents in the index."

    # Extract node IDs
    node_ids = list(nodes_dict.keys())

    count = min(count, len(node_ids))
    random_node_ids = random.sample(node_ids, count)
    return [index.docstore.get_node(node_id).text for node_id in random_node_ids]

# Main function
@hydra_runner(config_path="conf", config_name="rag_generating")
def main(cfg) -> None:
    
    # Load LLM
    logging.info("Loading LLM.")
    model_path = cfg.generating.llm.model_path
    cfg.generating.inference.tokens_to_generate = 912
    if cfg.generating.llm.model_type == "gpt":
        Settings.llm = NeMoGPTLLM(model_path=model_path, cfg=cfg)
    else:
        raise ValueError("Currently RAG pipeline supports 'gpt' for LLM models.")

    dummy = Settings.llm.complete("dummy")
    print("Dummy LLM:", dummy)
    # Load embedder
    logging.info("Loading embedder.")
    embed_model_path = cfg.indexing.embedder.model_path
    if cfg.indexing.embedder.model_type == "bert":
        Settings.embed_model = NeMoBertEmbeddings(model_path=embed_model_path, cfg=cfg)
    else:
        raise ValueError("Currently RAG pipeline supports 'bert' for embeddings models.")

    Settings.llm = None
    # Load index from disk
    logging.info("Loading index from disk.")
    index_path = cfg.indexing.index_path
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # Prepare data folder
    data = []
    output_data_folder = cfg.generating.output_dir
    os.makedirs(output_data_folder, exist_ok=True)
    output_data_file = os.path.join(output_data_folder, f"{cfg.generating.prefix}_data.jsonl")

    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=cfg.generating.top_k)
    
    print("Start Generate Retrieval Data!")
    for i in tqdm.tqdm(range(cfg.generating.num_sample)):
        neg_docs = []
        pos_doc = None

        random_doc = get_random_document_from_index(index)

        prompt = QUERY_PROMPT_TEMPLATE.format(document=random_doc)
        pred_query = Settings.llm.complete(prompt)
        pred_query = str(pred_query.text)

        nodes = retriever.retrieve(pred_query)

        for node in nodes:
            neg_doc = node.text
            filter_prompt = FILTER_PROMPT_TEMPLATTE.format(document=neg_doc, query=pred_query)
            resp = Settings.llm.complete(filter_prompt)
            resp = str(resp.text)
            if "false" in resp.lower():
                neg_docs.append(neg_doc)
            else:
                if pos_doc is None:
                    pos_doc = neg_doc

        # Add random hard negative samples
        random_docs = get_random_documents_from_index(index, cfg.generating.num_random)
        neg_docs.extend(random_docs)

        record = {"query": pred_query, "pos_doc": pos_doc, "neg_doc": neg_docs}
        data.append(record)

    with open(output_data_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    print("Save Generate Retrieval Data!", output_data_file)

if __name__ == "__main__":
    
    main()