NEMO_PATH=/home/aleksandraa/nemo

## Here we shuffle Yago phrases, but keep groups of 4 consequent phrases together.
## This is done in order to have examples of phrases with similar beginning in the same chunk that will serve as custom vocabulary
awk '{printf("%s%s",$0,(NR%4==0)?"\n":"\0")}' < custom_dict.txt > 1
sort -R 1 > 2
tr "\0" "\n" < 2 > custom_dict_shuffled.txt

## nemo_asr_set_sample_buckets1_4.json is a simplified manifest file. 
## It just serves as a source of sentences in which we insert misspelled phrases to generate synthetic data for the model.
## Its format is like this:
## {"text": "sixty nine belmont", "pred_text": "sixty nine belmont"}
## {"text": "for the european union", "pred_text": "for the european union"}
## {"text": "move to the third case", "pred_text": "move to the third case"}
## {"text": "much i yield the floor", "pred_text": "much i yield the floor"}

awk '(NR % 17 == 0){print $0}' < nemo_asr_set_sample_buckets1_4.json > nemo_asr_set_sample.json

mkdir examples

## We generate training data in the following way:
##   1. We generate many random samples of Yago phrases (size=5000) to serve as custom vocabularies
##   2. To get positive examples: for each Yago phrase from a vocabulary we take random sentence from nemo_asr_set_sample.json and insert misspelled phrase at random place in the sentence.
##      Then we index custom vocabulary and find top 10 candidates for our new sentence. The target is the index of the correct candidate.
##   3. To get negative examples: we just take random sentence from nemo_asr_set_sample.json, and find top 10 candidates - they are all negative, target is 0.
head -n 1500000 custom_dict_shuffled.txt | split -l 5000

for letter2 in "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" #"m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z"
do
    for letter3 in "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z"
    do
        part=x${letter2}${letter3}
        python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_corpora_after_alignment.py \
          --mode=index_by_vocab \
          --alignment_filename=${part} \
          --vocab_filename=replacement_vocab_filt.txt \
          --out_filename=index.txt

        python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_training_examples_for_bert.py \
          --index_name=index.txt \
          --input_manifest=nemo_asr_set_sample.json \
          --input_vocab=${part} \
          --output_name=examples/examples.${part}.txt
    done
done

DATASET=sent_3m_half_neg

mkdir ../bert/datasets/${DATASET}

cat examples/examples.*.txt > ../bert/datasets/${DATASET}/all.tsv
sort -R ../bert/datasets/${DATASET}/all.tsv > ../bert/datasets/${DATASET}/all_shuffle.tsv
tail -n 30000 ../bert/datasets/${DATASET}/all_shuffle.tsv > ../bert/datasets/${DATASET}/valid.tsv
head -n 2900000 ../bert/datasets/${DATASET}/all_shuffle.tsv > ../bert/datasets/${DATASET}/train.tsv

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

