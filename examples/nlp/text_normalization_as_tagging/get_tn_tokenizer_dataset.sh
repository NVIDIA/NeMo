NEMO_PATH=/home/aleksandraa/nemo

## corpus language
CORPUS_LANG=en
WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

## names of working subfolders
CORPUS_DIR=${WORK_DIR}/corpus
ALIGNMENT_DIR=${WORK_DIR}/alignment

## We now have a large collection of ITN phrase conversions that we know how to tag.
## Once again we loop through the Google TN dataset and create tag-labeled datasets, containing full sentences.
## If a sentence contains something that we do not know how to tag, we discard the whole sentence.
for subset in "dev" "train"
do
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
      --mode=get_labeled_corpus_for_tn_tokenizer \
      --giza_dir=${ALIGNMENT_DIR} \
      --alignment_filename=itn.select.out \
      --data_dir=${CORPUS_DIR}/${subset} \
      --vocab_filename="" \
      --out_filename=${CORPUS_DIR}/${subset}.tn_tokenizer.labeled \
      --lang=${CORPUS_LANG}
done


python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/get_label_vocab.py \
  --train_filename=${CORPUS_DIR}/train.tn_tokenizer.labeled \
  --dev_filename=${CORPUS_DIR}/dev.tn_tokenizer.labeled \
  --out_filename=${CORPUS_DIR}/tn_tokenizer_label_map.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
  --filename=${CORPUS_DIR}/dev.tn_tokenizer.labeled \
  --max_count=5000

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
  --filename=${CORPUS_DIR}/train.tn_tokenizer.labeled \
  --max_count=200000

mkdir datasets

DATASET=${WORK_DIR}/datasets/tn_tokenizer_sample_200k_200k
mkdir $DATASET
cat ${CORPUS_DIR}/train.tn_tokenizer.labeled.sample_200000 > ${DATASET}/train.tsv
head -n 200000 ${CORPUS_DIR}/train.tn_tokenizer.labeled.rest_200000 >> ${DATASET}/train.tsv
cat ${CORPUS_DIR}/dev.tn_tokenizer.labeled.sample_5000 > ${DATASET}/valid.tsv
head -n 12000 ${CORPUS_DIR}/dev.tn_tokenizer.labeled.rest_5000 >> ${DATASET}/valid.tsv
cp ${DATASET}/valid.tsv ${DATASET}/test.tsv
cp ${CORPUS_DIR}/tn_tokenizer_label_map.txt ${DATASET}/tn_tokenizer_label_map.txt
cp ${CORPUS_DIR}/semiotic_classes.txt ${DATASET}/semiotic_classes.txt
