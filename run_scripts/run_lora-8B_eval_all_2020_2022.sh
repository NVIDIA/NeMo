echo "****** STARTING ******" \
; echo "------------------" \
; BASE_MODEL="/home/hshin/projects/llmservice_modelalignment_ptune/checkpoints/8B/megatron_gpt_8b_tp4_pp1.nemo" \
; LORA_MODEL_DIR="/home/hshin/results/sec_sft" \
; DATA_DIR="/home/hshin/projects/llmservice_modelalignment_ptune/datasets/sec_qna_jsonls" \
; cd /home/hshin/workspace/NeMo \
; export PYTHONPATH="/home/hshin/workspace/NeMo/.:${PYTHONPATH}" \
; EVAL_MODEL_NAME=${EVAL_MODEL_NAME_INPUT}
; python examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
   model.restore_from_path=${BASE_MODEL} \
   model.peft.restore_from_path=/results/sec_sft/${EVAL_MODEL_NAME}/checkpoints/${EVAL_MODEL_NAME}.nemo \
   trainer.devices=4 \
   trainer.num_nodes=1 \
   trainer.precision=bf16 \
   model.tensor_model_parallel_size=4 \
   model.data.test_ds.file_names=[${DATA_DIR}/sec_qna_2020-2022_train_num-200-shuffle_val-test_input-outupt.jsonl] \
   model.data.test_ds.names=[sec_qna_2020-2022_val-test] \
   model.data.test_ds.global_batch_size=4 \
   model.data.test_ds.micro_batch_size=4 \
   model.data.test_ds.tokens_to_generate=30 \
   model.data.test_ds.write_predictions_to_file=True \
   model.data.test_ds.output_file_path_prefix=${EVAL_MODEL_NAME}.predictions \
   model.data.test_ds.label_key='output' \
   model.data.test_ds.truncation_field='input' \
   model.data.test_ds.prompt_template='{input} {output}' \
   inference.greedy=True \
   ++inference.verbose=True \
&& \
python run_scripts/parse_eval_answers_and_labels.py \
   --input_filename=${EVAL_MODEL_NAME}.predictions_test_sec_qna_2020-2022_val-test_inputs_preds_labels.jsonl \
&& \
cd /home/hshin/workspace/sec \
; python evaluate_answers.py \
   --filename=${EVAL_MODEL_NAME}.predictions_test_sec_qna_2020-2022_val-test_inputs_preds_labels.jsonl-ang_only.csv > \
   ${EVAL_MODEL_NAME}.predictions_test_sec_qna_2020-2022_val-test_inputs_preds_labels_ACCF1.txt
