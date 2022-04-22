DATA_PATH=/home/aleksandraa/data/tn_data/en_pipeline8/datasets
NEMO_PATH=/home/aleksandraa/nemo
PRETRAINED_MODEL=./nemo_experiments/training.nemo

export TOKENIZERS_PARALLELISM=false

## run inference on default Google Dataset test
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  inference.from_file=${DATA_PATH}/test.input \
  inference.out_file=./final_test.output \
  model.max_sequence_len=1024 #\
  inference.batch_size=128

## run inference on "hard" test (sample of at least 1000 examples of each semiotic class)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  inference.from_file=${DATA_PATH}/test1000.input \
  inference.out_file=./final_test1000.output \
  model.max_sequence_len=1024 \
  inference.batch_size=128

## compare inference results to the reference
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/eval.py \
  --reference_file=${DATA_PATH}/test.labeled \
  --inference_file=final_test.output \
  > final_test.report

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/eval.py \
  --reference_file=${DATA_PATH}/test1000.labeled \
  --inference_file=final_test1000.output \
  --print_other_errors \
  > final_test1000.report

## compare inference results to the reference, get separate report per semiotic class
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/eval_per_class.py \
  --reference_file=${DATA_PATH}/test.labeled \
  --inference_file=final_test.output \
  --output_file=per_class.report

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/utils/eval_per_class.py \
  --reference_file=${DATA_PATH}/test1000.labeled \
  --inference_file=final_test1000.output \
  --output_file=per_class1000.report
