# Repeat the below script continuously to evaluate the model on the test set
EXP_DIRS=(
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/CodeCompare"
    "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/CodeCompare"
)

EXP_NAMES=(
    "oldcode_dac_speakerid_op"
    "oldcode_dac_speakerid"
    "oldcode_encodec_speakerid"
    "oldcode_encodec_speakerid_noctc"
    "oldcode_ctc_0.05_parallel"
    "correct_ctc_dac_speakerid_newcode_step0_scale0.1"
)
# TEST_DS="/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_OnlyHifi_dac_test_speakerid.json"
TEST_DSS=(
    "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_dac_test_speakerid_op.json"
    "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_dac_test_speakerid.json"
    "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_encodec_test_speakerid.json"
    "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_encodec_test_speakerid.json"
    "/datap/misc/speechllm_codecdatasets/manifests/LRH_encodec_test.json"
    "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_dac_test_speakerid.json"
)

CODEC_MODEL_TYPES=(
    "dac"
    "dac"
    "encodec"
    "encodec"
    "encodec"
    "dac"
)

# Repeat whole thing 10 times

for ((j=0; j<10; j++)); do

for ((i=0; i<${#EXP_NAMES[@]}; i++)); do

EXP_DIR=${EXP_DIRS[i]}
EXP_NAME=${EXP_NAMES[i]}
TEST_DS=${TEST_DSS[i]}
CODEC_MODEL_TYPE=${CODEC_MODEL_TYPES[i]}

# if codec model type is dac, then set the codec fps to 100
if [ "$CODEC_MODEL_TYPE" = "dac" ]; then
    CODEC_FPS=86
    CODEC_MODEL_CODEBOOKS=9
else
    CODEC_FPS=75
    CODEC_MODEL_CODEBOOKS=8
fi

CHECKPOINT_PATH=$EXP_DIR/$EXP_NAME/p_tuning_squad_t5/checkpoints/*last.ckpt


LOCAL_CKPT_DIR="/datap/misc/temp_checkpoints"

scp pneekhara@selene-login:$CHECKPOINT_PATH $LOCAL_CKPT_DIR

# Read name of the checkpoint file in LOCAL_CKPT_DIR
CHECKPOINT_FILE=$(ls $LOCAL_CKPT_DIR | grep "last.ckpt")
# Take 1st line of the above output and first word of that line
CHECKPOINT_FILE=$(echo $CHECKPOINT_FILE | head -n 1 | awk '{print $1;}')

# Get the iter number after "step=" in the checkpoint file name
CHECKPOINT_ITER=$(echo $CHECKPOINT_FILE | sed -e 's/.*step=\([0-9]*\).*/\1/')

echo "Checkpoint file: $CHECKPOINT_FILE"
echo "Checkpoint iter: $CHECKPOINT_ITER"

# New checkpoint filename is EXP_NAME + CHECKPOINT_ITER .ckpt
NEW_CHECKPOINT_FILE=$EXP_NAME"_step"$CHECKPOINT_ITER".ckpt"
NEW_EXP_NAME=$EXP_NAME"_step"$CHECKPOINT_ITER

echo "New checkpoint file: $NEW_CHECKPOINT_FILE"

# rename the checkpoint file
echo "mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE

read -r -d '' cmd <<EOF
python examples/nlp/language_modeling/megatron_t5_speechlm_sft_inference.py \
--config-name=megatron_t5_speechlm_inference.yaml \
name=$NEW_EXP_NAME \
model.data.test_ds='["$TEST_DS"]' \
model.data.train_task=all \
+model.freeze_model=False \
model.data.max_seq_length=1536 \
model.max_inference_timesteps=1500 \
+model.data.context_duration_min=2.9 \
+model.data.context_duration_max=2.9 \
+model.data.context_pattern=parallel \
model.top_k=80 \
model.temperature=0.9 \
exp_manager.exp_dir=/datap/misc/gpt_local_multitask_experiments/AutomatedEvalFresh \
model.data.sup_data_path=/datap/misc/librittscodec/codec \
model.global_batch_size=2 \
model.micro_batch_size=2 \
model.data.speech_offset=30128 \
+model.data.num_speech_codebooks=$CODEC_MODEL_CODEBOOKS \
+model.data.codebook_fps=$CODEC_FPS \
+model.codecmodel_type=$CODEC_MODEL_TYPE \
+model.codecmodel_path=/datap/misc/Checkpoints/dac/weights_44khz_8kbps_0.0.1.pth \
+model.data.lm_vocab_size=30000 \
trainer.devices=1 \
trainer.precision=32 \
model.language_model_path=/datap/misc/Checkpoints/megatron_t5_220m/tp1_pp1/megatron_t5_expanded_vocab_posemb1536_220m.nemo \
model.seq_pattern=delay_parallel \
checkpoint_path="$LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE" \
model.speech_head_type=linear
EOF

echo "Running command: $cmd"

eval $cmd

# Remove the checkpoint file
echo "rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE

sleep 2m # sleep for 2 minutes

done

done
