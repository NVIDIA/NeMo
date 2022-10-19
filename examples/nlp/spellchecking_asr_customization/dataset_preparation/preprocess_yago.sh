NEMO_PATH=/home/aleksandraa/nemo

## download yagoTypes.tsv from https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/

awk 'BEGIN {FS="\t"} {print $2}' < yagoTypes.tsv | sort -u > yago.uniq
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/preprocess_yago.py --input_name yago.uniq --output_name yago.uniq2 --vocab_name yago.vocab.txt
sort -u yago.uniq2 > yago.uniq3
