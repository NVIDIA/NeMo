import torch
from transformers import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.file_utils import cached_path
import argparse

state_dict_mappings = {
    'gamma': 'weight',
    'beta': 'bias',
    'bert.encoder.layer': 'encoder.layers',
    'bert.embeddings.word_embeddings.weight': 'embedding_layer.word_embedding.'
    'weight',
    'bert.embeddings.position_embeddings.weight': 'embedding_layer.'
    'position_embedding.weight',
    'bert.embeddings.token_type_embeddings.weight': 'embedding_layer.token_'
    'type_embedding.weight',
    'bert.embeddings.LayerNorm.weight': 'embedding_layer.layer_norm.weight',
    'bert.embeddings.LayerNorm.bias': 'embedding_layer.layer_norm.bias',
    'attention.self.query.weight': 'first_sub_layer.query_net.weight',
    'attention.self.query.bias': 'first_sub_layer.query_net.bias',
    'attention.self.key.weight': 'first_sub_layer.key_net.weight',
    'attention.self.key.bias': 'first_sub_layer.key_net.bias',
    'attention.self.value.weight': 'first_sub_layer.value_net.weight',
    'attention.self.value.bias': 'first_sub_layer.value_net.bias',
    'attention.output.dense.weight': 'first_sub_layer.out_projection.weight',
    'attention.output.dense.bias': 'first_sub_layer.out_projection.bias',
    'attention.output.LayerNorm.weight': 'first_sub_layer.layer_norm.weight',
    'attention.output.LayerNorm.bias': 'first_sub_layer.layer_norm.bias',
    'intermediate.dense.weight': 'second_sub_layer.dense_in.weight',
    'intermediate.dense.bias': 'second_sub_layer.dense_in.bias',
    'output.dense.weight': 'second_sub_layer.dense_out.weight',
    'output.dense.bias': 'second_sub_layer.dense_out.bias',
    'output.LayerNorm.weight': 'second_sub_layer.layer_norm.weight',
    'output.LayerNorm.bias': 'second_sub_layer.layer_norm.bias'
}

decoder_keys = [
    'embedding_layer.token_embedding.weight',
    'embedding_layer.position_embedding.weight',
    'embedding_layer.token_type_embedding.weight',
    'embedding_layer.layer_norm.weight', 'embedding_layer.layer_norm.bias'
]

parser = argparse.ArgumentParser(description="BERT parameters to decoder")
parser.add_argument("--model_name", default="bert-base-uncased", type=str)
parser.add_argument("--save_to", default="", type=str)

args = parser.parse_args()

path = cached_path(BERT_PRETRAINED_MODEL_ARCHIVE_MAP[args.model_name])
weights_bert = torch.load(path)
bert_keys = list(weights_bert.keys())

nemo_bert_mapping = {}

# Map the keys to the nemo module
for key in weights_bert.keys():
    new_key = key

    for keys_to_map, mapped_key in state_dict_mappings.items():
        if keys_to_map in new_key:
            new_key = new_key.replace(keys_to_map, mapped_key)

    nemo_bert_mapping[key] = new_key

decoder_from_bert = {}

for i in range(5):
    decoder_from_bert[decoder_keys[i]] = bert_keys[i]

cur_layer = 0
for i in range(5, len(bert_keys)):
    key = nemo_bert_mapping[bert_keys[i]]
    if ("pooler" not in key) and ("cls" not in key):
        tmp = key.split(".")
        cur_layer = int(tmp[2])
        if "first" in key:
            key_first = ".".join(
                ["decoder", "layers", str(cur_layer)] + tmp[3:])
            key_second = ".".join(
                ["decoder", "layers",
                 str(cur_layer), "second_sub_layer"] + tmp[4:])
            decoder_from_bert[key_first] = bert_keys[i]
            decoder_from_bert[key_second] = bert_keys[i]
        elif "second" in key:
            key_third = ".".join(
                ["decoder", "layers",
                 str(cur_layer), "third_sub_layer"] + tmp[4:])
            decoder_from_bert[key_third] = bert_keys[i]

new_decoder_weights = {}
for key in decoder_from_bert.keys():
    new_decoder_weights[key] = weights_bert[decoder_from_bert[key]]

# Add zeros to make vocab_size divisible by 8 for fast training in
# mixed precision
vocab_size, d_model = new_decoder_weights[
    "embedding_layer.token_embedding.weight"].size()
tokens_to_add = 8 - vocab_size % 8
zeros = torch.zeros((tokens_to_add, d_model)).to(device="cpu")

tmp = torch.cat(
    (new_decoder_weights['embedding_layer.token_embedding.weight'], zeros))

new_decoder_weights['embedding_layer.token_embedding.weight'] = tmp
torch.save(new_decoder_weights, args.save_to + args.model_name + "_decoder.pt")
