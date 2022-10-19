NEMO_PATH=/home/aleksandraa/nemo

ALIGNMENT_DIR=align
## path to GIZA++ and mkcls binaries
GIZA_BIN_DIR=/home/aleksandraa/programs/giza-pp/GIZA++-v2
MCKLS_BINARY=/home/aleksandraa/programs/giza-pp/mkcls-v2/mkcls

mkdir ${ALIGNMENT_DIR}
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_input_for_giza.py \
  --input_manifest pred_ctc.all.json \
  --output_name giza_input.txt \
  --out_dir=${ALIGNMENT_DIR} \
  --giza_dir=${GIZA_BIN_DIR} \
  --mckls_binary=${MCKLS_BINARY}

awk 'BEGIN {FS="\t"} {print $1}' < giza_input.txt > ${ALIGNMENT_DIR}/src
awk 'BEGIN {FS="\t"} {print $2}' < giza_input.txt > ${ALIGNMENT_DIR}/dst
chmod +x ${ALIGNMENT_DIR}/run.sh

## Run Giza++ alignment
cd ${ALIGNMENT_DIR}
./run.sh
cd ..

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/extract_giza_alignments.py \
  --giza_dir=${ALIGNMENT_DIR} \
  --out_filename=align.out \
  --giza_suffix=A3.final

## !!!   align2.out has src=reference, dst=misspell
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab \
  --alignment_filename=${ALIGNMENT_DIR}/align2.out \
  --vocab_filename=replacement_vocab_full.txt \
  --out_filename=""

awk 'BEGIN {FS="\t"}($3 / $4 > 0.005){print $0}' < replacement_vocab_full.txt > replacement_vocab_filt.txt
awk '($1 == "good:"){print $0}' < ${ALIGNMENT_DIR}/align2.out | sort > custom_dict.txt
