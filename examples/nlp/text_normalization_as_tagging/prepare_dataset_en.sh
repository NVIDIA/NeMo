#!/bin/bash

WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

## corpus language
CORPUS_LANG=en

## path to NeMo repository
NEMO_PATH=/home/aleksandraa/nemo

## path to GIZA++ and mkcls binaries
GIZA_BIN_DIR=/home/aleksandraa/programs/giza-pp/GIZA++-v2
MCKLS_BINARY=/home/aleksandraa/programs/giza-pp/mkcls-v2/mkcls

## initial unzipped Google Text Normalization Dataset
GOOGLE_CORPUS_DIR=/home/aleksandraa/data/tn_data/en_with_types

## names of working subfolders
CORPUS_DIR=${WORK_DIR}/corpus
ALIGNMENT_DIR=${WORK_DIR}/alignment

## read the data and split it into train, dev, test
## files in test folder is truncated to match the default test dataset from the paper on Google TN Dataset
## option --add_test_full=true creates additional test_full folder, which is not truncated
python ${NEMO_PATH}/examples/nlp/duplex_text_normalization/data/data_split.py \
  --data_dir=${GOOGLE_CORPUS_DIR} \
  --output_dir=${CORPUS_DIR} \
  --lang=${CORPUS_LANG} \
  --add_test_full

## we need only output-00099-of-00100.tsv as the final test data
rm ${CORPUS_DIR}/test/output-00095-of-00100.tsv ${CORPUS_DIR}/test/output-00096-of-00100.tsv ${CORPUS_DIR}/test/output-00097-of-00100.tsv ${CORPUS_DIR}/test/output-00098-of-00100.tsv

mkdir ${ALIGNMENT_DIR}
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_for_alignment.py \
  --data_dir=${CORPUS_DIR} \
  --out_dir=${ALIGNMENT_DIR} \
  --giza_dir=${GIZA_BIN_DIR} \
  --mckls_binary=${MCKLS_BINARY} \
  --lang=${CORPUS_LANG}

##exclude punct class
rm -r ${ALIGNMENT_DIR}/punct

## for better giza alignments I mix in examples from other classes
## they will append to the tail of "src" and "dst" files and they will not have corresponding freqs in "freq" file
## all these appended lines will be skipped in the get_replacement_vocab step
for fn in "src" "dst"
do
    cat ${ALIGNMENT_DIR}/money/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} > ${ALIGNMENT_DIR}/money/${fn}.new

    cat ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/measure/${fn}.new

    cat ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/fraction/${fn}.new

    cat ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/decimal/${fn}.new

done

for c in "decimal" "fraction" "measure" "money"
do
    mv ${ALIGNMENT_DIR}/${c}/src.new ${ALIGNMENT_DIR}/${c}/src
    mv ${ALIGNMENT_DIR}/${c}/dst.new ${ALIGNMENT_DIR}/${c}/dst
done

for subfolder in ${ALIGNMENT_DIR}/*
do
    echo ${subfolder}
    chmod +x ${subfolder}/run.sh
done

for subfolder in ${ALIGNMENT_DIR}/*
do
    cd ${subfolder}
    ./run.sh &
done
wait

for subfolder in ${ALIGNMENT_DIR}/*
do
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/extract_giza_alignments.py \
      --mode=itn \
      --giza_dir=${subfolder} \
      --out_filename=itn.out \
      --giza_suffix=A3.final \
      --lang=en &
done
wait

for subfolder in ${ALIGNMENT_DIR}/*
do
    paste -d"\t" ${subfolder}/freq ${subfolder}/itn.out > ${subfolder}/itn.out2
    awk 'BEGIN {FS="\t"} match($3, " "){print $0}' < ${subfolder}/itn.out2 | sort -rn > ${subfolder}/itn.debug
done


python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/replacement_vocab_full.txt \
  --out_filename="" \
  --lang=${CORPUS_LANG}

grep -v "0__" replacement_vocab_full.txt.verbatim | head -n 108 > replacement_vocab_verbatim.txt
grep -v "0__" replacement_vocab_full.txt.time | head -n 148 > replacement_vocab_time.txt
grep -v "0__" replacement_vocab_full.txt.telephone | head -n 52 > replacement_vocab_telephone.txt
head -n 0 replacement_vocab_full.txt.plain > replacement_vocab_plain.txt
grep -v "0__" replacement_vocab_full.txt.ordinal | head -n 251 > replacement_vocab_ordinal.txt
grep -v "0__" replacement_vocab_full.txt.money | grep -v "a__" | head -n 532 > replacement_vocab_money.txt
grep -v "0__" replacement_vocab_full.txt.measure | head -n 488 > replacement_vocab_measure.txt
head -n 257 replacement_vocab_full.txt.letters > replacement_vocab_letters.txt
grep -v "0__" replacement_vocab_full.txt.fraction | head -n 169 > replacement_vocab_fraction.txt
head -n 276 replacement_vocab_full.txt.electronic > replacement_vocab_electronic.txt
head -n 73 replacement_vocab_full.txt.digit > replacement_vocab_digit.txt
grep -v "0__" replacement_vocab_full.txt.decimal | head -n 149 > replacement_vocab_decimal.txt
grep -v "0__" replacement_vocab_full.txt.date | grep -v "[0-9]-[0-9]" | grep -v "[0-9]\,[0-9]" | grep -v "[0-9]\.[0-9]" | grep -v "[0-9]\/[0-9]" | head -n 554 > replacement_vocab_date.txt
grep -v "0__" replacement_vocab_full.txt.cardinal | head -n 402 > replacement_vocab_cardinal.txt
head -n 137 replacement_vocab_full.txt.address > replacement_vocab_address.txt

cat replacement_vocab_address.txt \
  replacement_vocab_cardinal.txt \
  replacement_vocab_date.txt \
  replacement_vocab_decimal.txt \
  replacement_vocab_digit.txt \
  replacement_vocab_electronic.txt \
  replacement_vocab_fraction.txt \
  replacement_vocab_letters.txt \
  replacement_vocab_measure.txt \
  replacement_vocab_money.txt \
  replacement_vocab_ordinal.txt \
  replacement_vocab_plain.txt \
  replacement_vocab_telephone.txt \
  replacement_vocab_time.txt \
  replacement_vocab_verbatim.txt > replacement_vocab.select.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_after_alignment.py \
  --mode=filter_by_vocab \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/replacement_vocab.select.txt \
  --out_filename=itn.select.out \
  --lang=${CORPUS_LANG}

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_after_alignment.py \
  --mode=get_labeled_corpus \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.select.out \
  --data_dir=${CORPUS_DIR}/dev \
  --vocab_filename="" \
  --out_filename=${CORPUS_DIR}/dev.labeled \
  --lang=${CORPUS_LANG}

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_after_alignment.py \
  --mode=get_labeled_corpus \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.select.out \
  --data_dir=${CORPUS_DIR}/train \
  --vocab_filename="" \
  --out_filename=${CORPUS_DIR}/train.labeled \
  --lang=${CORPUS_LANG}

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/get_label_vocab.py \
  --train_filename=${CORPUS_DIR}/train.labeled \
  --dev_filename=${CORPUS_DIR}/dev.labeled \
  --out_filename=${CORPUS_DIR}/label_map.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/sample_each_label.py \
  --filename=${CORPUS_DIR}/dev.labeled \
  --max_count=10

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/sample_each_label.py \
  --filename=${CORPUS_DIR}/train.labeled \
  --max_count=500

echo "ADDRESS" > ${CORPUS_DIR}/semiotic_classes.txt
echo "CARDINAL" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "DATE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "DECIMAL" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "DIGIT" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "ELECTRONIC" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "FRACTION" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "LETTERS" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "MEASURE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "MONEY" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "ORDINAL" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "PLAIN" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "PUNCT" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "TELEPHONE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "TIME" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "VERBATIM" >> ${CORPUS_DIR}/semiotic_classes.txt

mkdir ${WORK_DIR}/datasets

cp ${CORPUS_DIR}/label_map.txt ${WORK_DIR}/datasets/label_map.txt
cp ${CORPUS_DIR}/semiotic_classes.txt ${WORK_DIR}/datasets/semiotic_classes.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/get_multi_reference_vocab.py \
  --data_dir=${CORPUS_DIR} \
  --out_filename=${CORPUS_DIR}/reference_vocab.txt

grep -P "[\d] m[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MEASURE" | grep -v -P "^MONEY" > ${CORPUS_DIR}/reference_vocab.bad1
grep -P "[\d] a[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MEASURE" > ${CORPUS_DIR}/reference_vocab.bad2
grep -P "[\d] b[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MEASURE" | grep -v -P "^MONEY" > ${CORPUS_DIR}/reference_vocab.bad3
grep -P "[\d] i[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MEASURE" > ${CORPUS_DIR}/reference_vocab.bad4
grep -P "[\d] i\-[\t]" ${CORPUS_DIR}/reference_vocab.txt > ${CORPUS_DIR}/reference_vocab.bad5
grep -P "[\d] us[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MONEY" | grep -v -P "TELEPHONE" > ${CORPUS_DIR}/reference_vocab.bad6
grep -P "[\d] u\.s\.[\t]" ${CORPUS_DIR}/reference_vocab.txt | grep -v -P "^MEASURE" | grep -v -P "^MONEY" > ${CORPUS_DIR}/reference_vocab.bad7
cat ${CORPUS_DIR}/reference_vocab.bad* > ${CORPUS_DIR}/reference_vocab.bad
grep -Fvxf ${CORPUS_DIR}/reference_vocab.bad ${CORPUS_DIR}/reference_vocab.txt > ${CORPUS_DIR}/reference_vocab.filt

## default test set (as in Google Dataset paper)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test \
  --reference_vocab=${CORPUS_DIR}/reference_vocab.filt \
  --output_file=${WORK_DIR}/datasets/test.labeled \
  --sampling_count=-1
awk 'BEGIN {FS="\t"}{print $1}' < ${WORK_DIR}/datasets/test.labeled > ${WORK_DIR}/datasets/test.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/test.labeled > ${WORK_DIR}/datasets/test.input_ref

## "hard" test set: at least 1000 examples per semiotic class
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test_full \
  --reference_vocab=${CORPUS_DIR}/reference_vocab.txt \
  --output_file=${WORK_DIR}/datasets/test1000.labeled \
  --sampling_count=1000
awk 'BEGIN {FS="\t"}{print $1}' < ${WORK_DIR}/datasets/test1000.labeled > ${WORK_DIR}/datasets/test1000.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/test1000.labeled > ${WORK_DIR}/datasets/test1000.input_ref


DATASET=${WORK_DIR}/datasets/itn_sample500k_rest1500k_select_vocab
mkdir $DATASET
cat ${CORPUS_DIR}/train.labeled.sample_500 > ${DATASET}/train.tsv
head -n 1500000 ${CORPUS_DIR}/train.labeled.rest_500 >> ${DATASET}/train.tsv
cat ${CORPUS_DIR}/dev.labeled.sample_10 > ${DATASET}/valid.tsv
head -n 12000 ${CORPUS_DIR}/dev.labeled.rest_10 >> ${DATASET}/valid.tsv
cp ${DATASET}/valid.tsv ${DATASET}/test.tsv

