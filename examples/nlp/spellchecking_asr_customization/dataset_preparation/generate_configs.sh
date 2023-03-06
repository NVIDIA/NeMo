## Generate necessary configs
## This is config of huawei-noah/TinyBERT_General_6L_768D    with type_vocab_size changed from 2 to 11 to support multiple separators
echo "{
  \"attention_probs_dropout_prob\": 0.1,
  \"cell\": {},
  \"model_type\": \"bert\",
  \"hidden_act\": \"gelu\",
  \"hidden_dropout_prob\": 0.1,
  \"hidden_size\": 768,
  \"initializer_range\": 0.02,
  \"intermediate_size\": 3072,
  \"max_position_embeddings\": 512,
  \"num_attention_heads\": 12,
  \"num_hidden_layers\": 6,
  \"pre_trained\": \"\",
  \"structure\": [],
  \"type_vocab_size\": 11,
  \"vocab_size\": 30522
}

" > ../bert/datasets/${DATASET}/config.json

## This is the set of possible target labels (0 - no replacements, 1-10 - replacement with candidate id)
echo "0
1
2
3
4
5
6
7
8
9
10
" > ../bert/datasets/${DATASET}/label_map.txt

## This is an auxiliary span labels needed for validation
echo "PLAIN
CUSTOM
" > ../bert/datasets/${DATASET}/semiotic_classes.txt

