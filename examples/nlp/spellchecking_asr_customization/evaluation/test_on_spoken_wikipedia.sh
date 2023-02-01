#!/bin/bash
NEMO_PATH=/home/aleksandraa/nemo

NEMO_PATH="/home/common/FBMF/studfbmf28/programs/NeMo"
DATA_DIR="/home/common/FBMF/studfbmf28/data/spoken_wikipedia"
PRETRAINED_MODEL=${DATA_DIR}/training.nemo
INPUT_DIR=${DATA_DIR}/english_prepared
OUTPUT_DIR=${DATA_DIR}/english_result
NGRAM_MAPPING=${DATA_DIR}/replacement_vocab_filt.txt
SUB_MISSPELLS=${DATA_DIR}/sub_misspells.txt

## Split ASR output transcriptions into shorter fragments to serve as ASR hypotheses for spellchecking model
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/create_custom_vocabs.py --folder ${INPUT_DIR} --processed_folder ${OUTPUT_DIR}/processed

## Split ASR output transcriptions into shorter fragments to serve as ASR hypotheses for spellchecking model
mkdir ${OUTPUT_DIR}/hypotheses
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/extract_asr_hypotheses.py \
  --manifest ${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  --folder ${OUTPUT_DIR}/hypotheses

## Prepare inputs for inference of neural customization spellchecking model
mkdir ${OUTPUT_DIR}/spellchecker_input
mkdir ${OUTPUT_DIR}/spellchecker_output
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/prepare_input_for_spellchecker_inference.py \
  --input_path ${DATA_DIR}/english_result/hypotheses/_OP_1..1341_CL_.txt \
  --input_vocab_path ${DATA_DIR}/english_prepared/vocabs/_OP_1..1341_CL_.custom.txt \
  --ngram_mapping ${DATA_DIR}/replacement_vocab_filt.txt \
  --sub_misspells_file ${DATA_DIR}/sub_misspells.txt \
  --output_path ${DATA_DIR}/english_result/spellchecker_input/_OP_1..1341_CL_.txt \
  --output_info_path ${DATA_DIR}/english_result/spellchecker_input/_OP_1..1341_CL_.info.txt

## Create filelist with input filenames
rm ${DATA_DIR}/filelist.txt
for i in {1..1341}
do
    echo ${DATA_DIR}/english_result/spellchecker_input/${i}.txt >> ${DATA_DIR}/filelist.txt
done

## Run inference with neural customization spellchecking model
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  model.max_sequence_len=512 \
  +inference.from_filelist=${DATA_DIR}/filelist.txt \
  +inference.output_folder=${OUTPUT_DIR}/spellchecker_output \
  inference.batch_size=16 \
  lang=en

## Postprocess and combine spellchecker results into a single manifest
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/update_transcription_with_spellchecker_results.py \
  --asr_hypotheses_folder ${OUTPUT_DIR}/hypotheses \
  --spellchecker_inputs_folder ${OUTPUT_DIR}/spellchecker_input \
  --spellchecker_results_folder ${OUTPUT_DIR}/spellchecker_output \
  --input_manifest ${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  --output_manifest ${OUTPUT_DIR}/manifests/manifest_corrected_dp.json \
  --min_cov 0.4 \
  --min_real_cov 0.8 \
  --min_dp_score_per_symbol -1.5 \
  --ngram_mappings ../replacement_vocab_filt.txt

## Check CER of spellchecker results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_corrected_dp.json \
  use_cer=True \
  only_score_manifest=True

## Check WER of spellchecker results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_corrected_dp.json \
  use_cer=False \
  only_score_manifest=True

## Check CER of baseline ASR results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  use_cer=True \
  only_score_manifest=True

## Check WER of baseline ASR results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  use_cer=False \
  only_score_manifest=True

## Perform error analysis and create "ideal" spellchecker results for comparison
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/analyze_custom_ref_vs_asr.py \
  --manifest ${OUTPUT_DIR}/manifests/manifest_corrected_dp.json \
  --vocab_dir ${INPUT_DIR}/vocabs \
  --input_dir ${OUTPUT_DIR}/spellchecker_input \
  --ngram_mappings replacement_vocab_filt.txt \
  --output_name ${OUTPUT_DIR}/analysis_ref_vs_asr.txt

## Check CER of "ideal" spellcheck results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/analysis_ref_vs_asr.txt.ideal_spellcheck \
  use_cer=True \
  only_score_manifest=True

## Check WER of "ideal" spellcheck results
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/analysis_ref_vs_asr.txt.ideal_spellcheck \
  use_cer=False \
  only_score_manifest=True

