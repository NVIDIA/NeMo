## Streaming ASR with Speaker Tagging

### Install requirements

You need to install the following packages 

```
kenlm
arpa
meeteval
requests
simplejson
pydiardecode @ git+https://github.com/tango4j/pydiardecode@main
```

## Prepare three models
### 1. Streaming Sortformer Diarizer Model

- Streaming Sortformer: Download no feature normalization sortformer diarizer model for streaming 
https://drive.google.com/file/d/1UxnppMNn8ZmtrSOPPkGTLsTwGhZAPxcr/view?usp=sharing

### 2. Cache-aware Streaming ASR Model

- Download from the following link 
https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi

### 3. ARPA language model (n-gram)

```
mkdir -p arpa_model
cd arpa_model
wget https://kaldi-asr.org/models/5/4gram_small.arpa.gz
gunzip 4gram_small.arpa.gz
```
## Launcher Script


```
branch_name="sortformer_pr01_fifo_memory"
export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH
NEMO_ROOT=/home/taejinp/projects/$branch_name/NeMo
BASEPATH=${NEMO_ROOT}/examples/speaker_tasks/diarization

ASR_MODEL=/disk_a/models/streaming_asr/stt_en_fastconformer_hybrid_large_streaming_multi.nemo
MANIFEST_FILE=/disk_a/datasets/above4spk_finetune_datasets/sortformer_manifest_evalset_v3/chaes/en_0638.json
DIAR_MODEL_PATH=/disk_a/models/sortformer_diarization/stream_finetuned/im382no-normNA-mem1_epoch6-18.nemo
ARPA_LANGUAGE_MODEL=/disk_a/models/language_models/arpa_models/4gram_small.arpa

export CUDA_VISIBLE_DEVICES="0"

NUM_WORKERS=0
TEST_BATCHSIZE=1

MEMORY_FRAME_LENGTH=188 
DIARIZATION_STEP_FRAME_LENGTH=0 # CHUNK LEN
DIARIZATION_LEFT_CONTEXT_SIZE=0 # LEFT CONTEXT SIZE
DIARIZATION_RIGHT_CONTEXT_SIZE=0 # RIGHT CONTEXT SIZE
DIARIZATION_FIFO_QUEUE_SIZE=188 # 188 # FIFO QUEUE SIZE

PRINT_PATH=/home/taejinp/projects/$branch_name/print_script.sh
BEAM_SEARCH_ENABLED=true

python $NEMO_ROOT/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer_with_diarization.py \
    asr_model=$ASR_MODEL \
    diar_model_path=$DIAR_MODEL_PATH \
    arpa_language_model=$ARPA_LANGUAGE_MODEL \
    manifest_file=$MANIFEST_FILE \
    use_amp=true \
    debug_mode=false \
    bypass_postprocessing=true \
    eval_mode=true \
    num_workers=$NUM_WORKERS \
    batch_size=$TEST_BATCHSIZE \
    mem_len=$MEMORY_FRAME_LENGTH \
    step_len=$DIARIZATION_STEP_FRAME_LENGTH \
    fifo_len=$DIARIZATION_FIFO_QUEUE_SIZE \
    step_left_context=$DIARIZATION_LEFT_CONTEXT_SIZE \
    step_right_context=$DIARIZATION_RIGHT_CONTEXT_SIZE \
    collar=$COLLAR \
    print_path=$PRINT_PATH \
    colored_text=true \
    log=true \
    beam_prune_logp=-20 \
    alpha=1.0 \
    beta=0.01 \
    beam_width=9 \
    word_window=50 \
    beam_search_enabled=$BEAM_SEARCH_ENABLED \
    parallel_chunk_word_len=175 
```

## How to display the Transcription with Speaker Tagging ?

- Go to $PRINT_PATH folder and run the following script in your ANOTHER separate tmux screen.

```
print_script_a="script -q -c 'cat print_script.sh' /dev/null"; watch  -n 0.1 --color "$print_script_a"
```
