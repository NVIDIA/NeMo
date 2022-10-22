WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

DATA_PATH=${WORK_DIR}/datasets

NEMO_PATH=/home/aleksandraa/nemo
PRETRAINED_MODEL_TN_TOKENIZER=./nemo_experiments/tn_tokenizer.nemo
PRETRAINED_MODEL_TN_TAGGER=./nemo_experiments/tn_tagger.nemo

export TOKENIZERS_PARALLELISM=false

## run tn tokenizer inference on default Google Dataset test
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL_TN_TOKENIZER} \
  inference.from_file=${DATA_PATH}/tn_test.input \
  inference.out_file=./tn_test.tn_tokenizer.output \
  model.max_sequence_len=1024 \
  inference.batch_size=128 \
  lang=en

## run tn tokenizer inference on "hard" test (sample of at least 1000 examples of each semiotic class)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL_TN_TOKENIZER} \
  inference.from_file=${DATA_PATH}/tn_test1000.input \
  inference.out_file=./tn_test1000.tn_tokenizer.output \
  model.max_sequence_len=1024 \
  inference.batch_size=128 \
  lang=en

## !!!run tokenize.py for both files
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/tokenize.py tn_test.tn_tokenizer.output tn_test.tokenized_input
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/tokenize.py tn_test1000.tn_tokenizer.output tn_test1000.tokenized_input

## run tn inference on default Google Dataset test
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL_TN_TAGGER} \
  inference.from_file=./tn_test.tokenized_input \
  inference.out_file=./tn_test.output \
  model.max_sequence_len=1024 \
  inference.batch_size=128 \
  lang=en

## run tn inference on "hard" test (sample of at least 1000 examples of each semiotic class)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL_TN_TAGGER} \
  inference.from_file=./tn_test1000.tokenized_input \
  inference.out_file=./tn_test1000.output \
  model.max_sequence_len=1024 \
  inference.batch_size=128 \
  lang=en

## compare inference results to the reference
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval.py \
  --reference_file=${DATA_PATH}/tn_test.labeled \
  --inference_file=tn_test.output \
  --print_other_errors \
  > tn_test.report

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval.py \
  --reference_file=${DATA_PATH}/tn_test1000.labeled \
  --inference_file=tn_test1000.output \
  --print_other_errors \
  > tn_test1000.report

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval_per_class.py \
  --reference_file=${DATA_PATH}/tn_test.labeled \
  --inference_file=tn_test.output \
  --output_file=tn_per_class.report

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval_per_class.py \
  --reference_file=${DATA_PATH}/tn_test1000.labeled \
  --inference_file=tn_test1000.output \
  --output_file=tn_per_class1000.report
