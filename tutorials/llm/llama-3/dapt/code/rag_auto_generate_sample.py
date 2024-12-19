# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import random
from typing import List

import tqdm
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from nemo.collections.nlp.models.rag.custom_bert_embedder import NeMoBertEmbeddings
from nemo.collections.nlp.models.rag.custom_gpt_llm import NeMoGPTLLM
from nemo.core.config import hydra_runner

QUERY_PROMPT_TEMPLATE = """You will be provided with a document or a passage. Your task is to generate a single, highly relevant and natural language query that aligns perfectly with the content of the document.
The query should:
    1. Reflect the main idea or a key detail from the document.
    2. Be concise, specific, and written in natural language.
    3. Be something a user might naturally ask to retrieve this document.
    4. **Write only the answer, do not repeat the instructions or document.**
 
 ## Given Document:
 {document}
 
 ## Predict Query:
 """
 
FILTER_PROMPT_TEMPLATE = """You will be provided with a document and a query.
Your task is to evaluate whether the content of the document is relevant to answering the query. 

Return "True" if the document contains information directly related to the query, and "False" if it does not.
**Provide only the answer: "True" or "False", without repeating the instructions, document, or query.**

## Given Query:
{query}

## Given Document:
{document}

## Is Relevant (True or False):
"""

# Helper functions
def get_random_document_from_index(index: BaseIndex) -> str:
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

def get_random_documents_from_index(index: BaseIndex, count: int) -> List[str]:
    
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
        pred_query = str(pred_query.text).split("## Predict Query:")[-1].strip()

        nodes = retriever.retrieve(pred_query)

        for node in nodes:
            neg_doc = node.text
            filter_prompt = FILTER_PROMPT_TEMPLATE.format(document=neg_doc, query=pred_query)
            resp = Settings.llm.complete(filter_prompt)
            resp = str(resp.text).split("## Is Relevant (True or False):")[-1].strip()
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