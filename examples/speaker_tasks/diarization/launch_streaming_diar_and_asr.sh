function check_path() {
    local this_path=$1
    ls $this_path || { echo "Path $this_path does not exist"; exit 1; }
}

### Step 1 Pull NeMo streaming mul-spk ASR branch https://github.com/NVIDIA/NeMo/tree/streaming_mulspk_asr
NEMO_BASE_PATH="<your path to NeMo/examples/speaker_tasks/diarization>"
NEMO_ENV_PATH="<your path to NeMo>"
# branch=_streaming_mulspk_asr
# NEMO_BASE_PATH=/home/taejinp/projects/$branch/NeMo/examples/speaker_tasks/diarization
# NEMO_ENV_PATH=/home/taejinp/projects/$branch/NeMo
export PYTHONPATH=$NEMO_ENV_PATH:$PYTHONPATH

### Step 2 setup your buffer directory
out_dir="<your path to buffer directory>"
# out_dir=/home/taejinp/projects/run_time/streaming_diar_output_univ
check_path $out_dir

mkdir -p $out_dir
pushd $NEMO_BASE_PATH

### Step 3 setup your diarization manifest file
### You can download wav files and rttm files for ch109 from https://drive.google.com/drive/folders/15jJNLdVcuEh-FmpL6CN9acrGpXZ_B5Fz?usp=sharing
### You can download sample diarization test audio from https://drive.google.com/drive/folders/15riQtFD2d_e2R08A7cGNp0be7ckois-Q?usp=drive_link
### The setup of diarization manifest files can be found here: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/datasets.html#data-preparation-for-inference
### Example online_diar_demo_01.json: {"audio_filepath": "<yourpath>/diarization_demo_wavs/citadel_ken.wav", "offset": 0.0, "duration": 0.0, "label": "infer", "text": "-", "num_speakers": 4, "rttm_filepath": null, "uem_filepath": null, "ctm_filepath": null}
test_manifest="<your path to diarization manifest file>"
# test_manifest="/home/taejinp/projects/data/diar_manifest_input/online_diar_demo_01.json" 
uniq_id='citadel_ken'
check_path $test_manifest

export CUDA_VISIBLE_DEVICES="0"

### Step 4 Specify VAD and ASR model path
### Download ASR model from https://drive.google.com/file/d/1BJgUeNPGqnVKJMaPOdjqxd66zoWHDkaU/view?usp=share_link
### Download VAD model from https://drive.google.com/file/d/1ab42CaYeTkuJSMsMsMLbSS9m5e1isJzx/view?usp=sharing
asr_model_path="<your path to ASR model>"
# asr_model_path="/home/taejinp/gdrive/model/ASR_models/Conformer-CTC-BPE_large_Riva_ASR_set_3.0_ep60.nemo"
check_path $asr_model_path
vad_model_path="<your path to VAD model>"
# vad_model_path="/disk_c/taejinp/gdrive/model/VAD_models/mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd.nemo"
check_path $vad_model_path

### Step 5 Run the following in `out_dir` path to display realtime online diarization + ASR script
### $ watch  -n 0.1 --color "script -q -c 'cat print_script.sh' /dev/null"

### Step 6. Launch the online diarization + ASR (simulated streaming) script
python $NEMO_BASE_PATH/clustering_diarizer/online_diar_with_asr_infer.py \
    diarizer.simulation_uniq_id=$uniq_id \
    diarizer.manifest_filepath=$test_manifest \
    diarizer.vad.model_path=$vad_model_path \
    diarizer.asr.parameters.word_ts_anchor_offset=0.2 \
    diarizer.asr.parameters.asr_based_vad_threshold=1.0 \
    diarizer.asr.parameters.enforce_real_time=False \
    diarizer.asr.model_path=$asr_model_path \
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
