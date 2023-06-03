# branch=_streaming_mulspk_asr
# NEMO_BASE_PATH=/home/taejinp/projects/$branch/NeMo/examples/speaker_tasks/diarization

### Step 1 Pull NeMo streaming mul-spk ASR branch https://github.com/NVIDIA/NeMo/tree/streaming_mulspk_asr
NEMO_BASE_PATH=~/projects/$branch/NeMo/examples/speaker_tasks/diarization

### Step 2 setup your buffer directory
out_dir=/home/taejinp/projects/run_time/streaming_diar_output_univ

mkdir -p $out_dir
pushd $NEMO_BASE_PATH

### Step 3 setup your diarization manifest file
### You can download wav files and rttm files for ch109 from https://drive.google.com/drive/folders/15jJNLdVcuEh-FmpL6CN9acrGpXZ_B5Fz?usp=sharing
### The setup of diarization manifest files can be found here: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/datasets.html#data-preparation-for-inference
test_manifest='/home/taejinp/projects/data/diar_manifest_input/ch109.short3.json'

export CUDA_VISIBLE_DEVICES="0"

### Step 4 Specify an ASR model path
### Download it from https://drive.google.com/file/d/1BJgUeNPGqnVKJMaPOdjqxd66zoWHDkaU/view?usp=share_link
model_path="~/gdrive/model/ASR_models/Conformer-CTC-BPE_large_Riva_ASR_set_3.0_ep60.nemo"

### Step 5 Run the following in `out_dir` path to display realtime online diarization + ASR script
### $ watch  -n 0.1 --color "script -q -c 'cat print_script.sh' /dev/null"

python $NEMO_BASE_PATH/clustering_diarizer/online_diar_with_asr_infer.py \
    diarizer.simulation_uniq_id='en_0638' \
    diarizer.manifest_filepath=$test_manifest \
    diarizer.asr.parameters.word_ts_anchor_offset=0.2 \
    diarizer.asr.parameters.asr_based_vad_threshold=1.0 \
    diarizer.asr.parameters.enforce_real_time=False \
    diarizer.asr.model_path=$model_path \
    diarizer.speaker_embeddings.model_path="titanet_large" \
    diarizer.out_dir=$out_dir \
    diarizer.oracle_vad=False \
    diarizer.collar=0.0\
    diarizer.ignore_overlap=False\
    diarizer.asr.parameters.streaming_simulation=True \
    diarizer.asr.parameters.punctuation_model_path=null \
    diarizer.speaker_embeddings.parameters.window_length_in_sec='[2.0,1.0,0.5]' \
    diarizer.speaker_embeddings.parameters.shift_length_in_sec='[1.0,0.5,0.25]' \
    diarizer.speaker_embeddings.parameters.multiscale_weights='[1,1,1]'\
    diarizer.clustering.parameters.history_buffer_size=150 \
    diarizer.clustering.parameters.current_buffer_size=200  \

