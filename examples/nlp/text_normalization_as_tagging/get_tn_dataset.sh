NEMO_PATH=/home/aleksandraa/nemo

## corpus language
CORPUS_LANG=en
WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

## names of working subfolders
CORPUS_DIR=${WORK_DIR}/corpus
ALIGNMENT_DIR=${WORK_DIR}/alignment

## The original test data from Google TN Dataset contains a single reference for each TN span.
## In order to take into account more than one acceptable variant,
## we prepare a dictionary of multiple possible references.
## The following script maps the whole input text of TN span to the list of different conversions that occurred
## with this input anywhere in the Google TN Dataset.
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py \
  --data_dir=${CORPUS_DIR} \
  --out_filename=${CORPUS_DIR}/tn_reference_vocab.txt \
  --tn_direction \
  --lang=${CORPUS_LANG}

## default test set (as in Google Dataset paper)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test \
  --reference_vocab=${CORPUS_DIR}/tn_reference_vocab.txt \
  --output_file=${WORK_DIR}/datasets/tn_test.labeled \
  --sampling_count=-1 \
  --tn_direction \
  --lang=${CORPUS_LANG}

awk 'BEGIN {FS="\t"}{print "<BOS> " $1}' < ${WORK_DIR}/datasets/tn_test.labeled > ${WORK_DIR}/datasets/tn_test.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/tn_test.labeled > ${WORK_DIR}/datasets/tn_test.input_ref

## "hard" test set: at least 1000 examples per semiotic class
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test_full \
  --reference_vocab=${CORPUS_DIR}/tn_reference_vocab.txt \
  --output_file=${WORK_DIR}/datasets/tn_test1000.labeled \
  --sampling_count=1000 \
  --tn_direction \
  --lang=${CORPUS_LANG}

awk 'BEGIN {FS="\t"}{print "<BOS> " $1}' < ${WORK_DIR}/datasets/tn_test1000.labeled > ${WORK_DIR}/datasets/tn_test1000.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/tn_test1000.labeled > ${WORK_DIR}/datasets/tn_test1000.input_ref

## Prepare dataset for TN
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab_tn \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/tn_replacement_vocab_full.txt \
  --out_filename="" \
  --lang=${CORPUS_LANG}


head -n 46 tn_replacement_vocab_full.txt.electronic > tn_replacement_vocab_electronic.txt
head -n 446 tn_replacement_vocab_full.txt.plain > tn_replacement_vocab_plain.txt
head -n 42 tn_replacement_vocab_full.txt.telephone > tn_replacement_vocab_telephone.txt

cat tn_replacement_vocab_full.txt.address \
  tn_replacement_vocab_full.txt.cardinal \
  tn_replacement_vocab_full.txt.date \
  tn_replacement_vocab_full.txt.decimal \
  tn_replacement_vocab_full.txt.digit \
  tn_replacement_vocab_full.txt.fraction \
  tn_replacement_vocab_full.txt.measure \
  tn_replacement_vocab_full.txt.money \
  tn_replacement_vocab_full.txt.ordinal \
  tn_replacement_vocab_full.txt.time \
  tn_replacement_vocab_full.txt.verbatim \
  tn_replacement_vocab_electronic.txt \
  tn_replacement_vocab_plain.txt \
  tn_replacement_vocab_telephone.txt > tn_replacement_vocab.select.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=filter_by_vocab_tn \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/tn_replacement_vocab.select.txt \
  --out_filename=tn.select.out \
  --lang=${CORPUS_LANG}

for subset in "train" "dev"
do
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
      --mode=get_labeled_corpus_tn \
      --giza_dir=${ALIGNMENT_DIR} \
      --alignment_filename=tn.select.out \
      --data_dir=${CORPUS_DIR}/${subset} \
      --vocab_filename="" \
      --out_filename=${CORPUS_DIR}/tn.${subset}.labeled \
      --lang=${CORPUS_LANG}
done

#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/get_label_vocab.py \
#  --train_filename=${CORPUS_DIR}/tn.train.labeled \
#  --dev_filename=${CORPUS_DIR}/tn.dev.labeled \
#  --out_filename=${CORPUS_DIR}/tn_label_map.txt

#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
#  --filename=${CORPUS_DIR}/tn.dev.labeled \
#  --max_count=100

#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
#  --filename=${CORPUS_DIR}/tn.train.labeled \
#  --max_count=5000

#DATASET=${WORK_DIR}/datasets/tn_sample_500k_500k
#mkdir $DATASET
#cat ${CORPUS_DIR}/tn.train.labeled.sample_5000 > ${DATASET}/train.tsv
#head -n 500000 ${CORPUS_DIR}/tn.train.labeled.rest_5000 >> ${DATASET}/train.tsv
#cat ${CORPUS_DIR}/tn.dev.labeled.sample_100 > ${DATASET}/valid.tsv
#head -n 15000 ${CORPUS_DIR}/tn.dev.labeled.rest_100 >> ${DATASET}/valid.tsv
#cp ${DATASET}/valid.tsv ${DATASET}/test.tsv
#cp ${CORPUS_DIR}/tn_label_map.txt ${DATASET}/tn_label_map.txt
#cp ${CORPUS_DIR}/semiotic_classes.txt ${DATASET}/semiotic_classes.txt

## remove some sentence that contains hieroglyph leading to token number mismatch
#grep -v "nervastella" ${DATASET}/train.tsv > ${DATASET}/train.tsv1
#mv ${DATASET}/train.tsv1 ${DATASET}/train.tsv
