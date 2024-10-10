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

import torch


def load_weights(load_path, save_path):
    """
    This function loads llama2 weights (hugging face) and converts it to a Dict format suitable for NeMo

    Args:
            load_path (str): Path to llama2 weights downlaoded from hugging face
            save_path (str): Path to save modified dictionary containing the weights.

    Returns:
            None

    """
    model_type = "llama2"
    for i in range(8):
        print(f"Snapshot {i}")
        state_dict = torch.load(f"{load_path}/consolidated.0{i}.pth")
        batch_dict = {}
        if model_type == "llama2":
            batch_dict['word_embeddings'] = state_dict['tok_embeddings.weight']
            batch_dict['output_layer'] = state_dict['output.weight']
        else:
            batch_dict['word_embeddings'] = state_dict['model']['embedding.word_embeddings.weight']
            batch_dict['output_layer'] = state_dict['model']['output_layer.weight']
        print("Embedding layer dimension: ", batch_dict['word_embeddings'].shape)
        print("Output layer dimension: ", batch_dict['output_layer'].shape)
        torch.save(batch_dict, f'{save_path}/embedding_{i}.pt')


def merge_embed(old_embd_path, new_embd_path, save_path):
    "Function to merge embeddings and convert back to hugging face format"
    model_type = "llama2"
    for i in range(8):
        print(f"Snapshot {i}")
        state_dict = torch.load(f"{old_embd_path}/consolidated.0{i}.pth")
        batch_dict = torch.load(f'{new_embd_path}/embedding_{i}.pt')
        if model_type == "llama2":
            state_dict['output.weight'] = batch_dict['output_layer']
            state_dict['tok_embeddings.weight'] = batch_dict['word_embeddings']
            print("embedding shape: ", state_dict['tok_embeddings.weight'].shape)
            print("output shape: ", state_dict['output.weight'].shape)
        #             state_dict['args'].padded_vocab_size = state_dict['model']['language_model']['embedding']['word_embeddings']['weight'].shape[0] * 8
        #             state_dict['args'].vocab_size = state_dict['args'].padded_vocab_size - 768
        #             print("vocab_size: ", state_dict['args'].vocab_size)
        #             print("padded_vocab_size: ", state_dict['args'].padded_vocab_size)
        else:
            state_dict['tok_embeddings.weight'] = batch_dict['word_embeddings']
            state_dict['output.weight'] = batch_dict['output_layer']
            print("embedding shape: ", state_dict['model']['embedding.word_embeddings.weight'].shape)
            print("output shape: ", state_dict['model']['output_layer.weight'].shape)
        #             state_dict['args'].padded_vocab_size = state_dict['model']['embedding.word_embeddings.weight'].shape[0] * 8
        #             state_dict['args'].vocab_size = state_dict['args'].padded_vocab_size - 768
        #             print("vocab_size: ", state_dict['args'].vocab_size)
        #             print("padded_vocab_size: ", state_dict['args'].padded_vocab_size)
        torch.save(state_dict, f"{save_path}/consolidated.0{i}.pth")
        print(f"Done merging snapshot {i}")
