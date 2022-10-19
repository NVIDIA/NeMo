#!/bin/bash
NEMO_PATH=/home/aleksandraa/nemo

INPUT_DIR=english
OUTPUT_DIR=english_result

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/create_custom_vocabs.py --folder ${INPUT_DIR}_prepared --processed_folder ${OUTPUT_DIR}/processed

## Apply Spelling Correction
mkdir ${OUTPUT_DIR}/hypotheses
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/extract_asr_hypotheses.py --manifest ${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json --folder ${OUTPUT_DIR}/hypotheses

PRETRAINED_MODEL=../bert/nemo_experiments/training_3000k_half_neg.nemo

mkdir ${OUTPUT_DIR}/spellchecker_input
mkdir ${OUTPUT_DIR}/spellchecker_output

for doc_id in {1..1332}
do
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/prepare_input_for_spellchecker_inference.py \
      --input_file ${OUTPUT_DIR}/hypotheses/${doc_id}.txt \
      --input_vocab ${INPUT_DIR}_prepared/vocabs/${doc_id}.custom.txt \
      --ngram_mapping replacement_vocab_filt.txt \
      --output_name ${OUTPUT_DIR}/spellchecker_input/${doc_id}.txt
done

for doc_id in {1..1332}
do
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
      pretrained_model=${PRETRAINED_MODEL} \
      inference.from_file=${OUTPUT_DIR}/spellchecker_input/${doc_id}.txt \
      inference.out_file=${OUTPUT_DIR}/spellchecker_output/${doc_id}.txt \
      inference.batch_size=256 \
      lang=en
done

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/update_transcription_with_spellchecker_results.py \
  --asr_hypotheses_folder english_result/hypotheses \
  --spellchecker_results_folder english_result/spellchecker_output \
  --input_manifest english_result/manifests/manifest_transcribed_metrics_filtered.json \
  --output_manifest english_result/manifests/manifest_corrected.json

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_corrected.json \
  use_cer=True \
  only_score_manifest=True

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_corrected.json \
  use_cer=False \
  only_score_manifest=True

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/analyze_errors.py --input_manifest ${OUTPUT_DIR}/manifests/manifest_corrected.json --output_file report.txt
