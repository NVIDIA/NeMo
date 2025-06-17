This guidance describes the new Lhotse Shar process for converting NeMo datasets to Lhotse Shar datasets, designed for
training and validating Magpie-TTS. This new version significantly reduces computation overhead by using rank-balanced
workloading and independent writing across parallel processes. It also separate the processes to CPU-only nodes and
GPU-only nodes accordingly.

## Creating New Lhotse Shar Data

The process involves four main steps:
1.  **Prepare Input Manifests (on CPU nodes):** Standardize the input NeMo manifests for each dataset.
2.  **Extend Manifests with Context Audio (on GPU nodes):** Enhance the NeMo manifests by adding context audio information.
3.  **Create Lhotse Shards (on CPU nodes):** Convert the extended NeMo manifests into Lhotse shards.
4.  **Extend Shards with Audio Codes (on GPU nodes):** Process the Lhotse shards to extract and include audio codes (audio codec extraction).

### Step 1: Prepare Input Manifests

This first step runs on **CPU nodes** and is responsible for standardizing the input NeMo manifests for each dataset. This may involve consolidating multiple input files or reformatting entries. It's a preparatory step to ensure the manifest is in the correct format for the subsequent stages.

*Note: The actual implementation for this step ([`prep_input_manifest.py`](https://gitlab-master.nvidia.com/xueyang/nemo-tts-artifacts-registry/-/blob/model_release_2505/model_release_2505/data_prep/hifitts2/prep_input_manifest_iad.py) in the internal scripts) is highly specific to the dataset and environment. Users should create their own script to prepare a standardized manifest file as input for Step 2.*

A crucial part of this step is to ensure the `speaker` field in the NeMo manifest conforms to the required format:
```python
def check_speaker_format(item: str):
    # enforce the format as example like "| Language:en Dataset:HiFiTTS Speaker:9136_other |".
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))
```

#### Checkout the Outputs of `hifitts2/prep_input_manifest.py`
```bash
$ tree -L 1 -P '*.json|*.txt' hifitts2/nemo_manifest/
hifitts2/nemo_manifest/
├── hifitts2_all_splits.json
├── hifitts2_dev_seen.json
├── hifitts2_dev_unseen.json
├── hifitts2_test_seen.json
├── hifitts2_test_unseen.json
├── hifitts2_train.json  # This is the standardized NeMo manifest used for the following steps.
├── manifest_empty_normalized_text_fields.json
├── manifest_librosa_error.json
├── manifest_mismatched_audio_duration.json
├── manifest_missing_asr_metrics.json
├── manifest_missing_audio_files.json
├── manifest_missing_required_fields.json
└── stats.txt  # This helps to understand the success and failure records.
```

### Step 2: Extend NeMo Manifest with Context Audio

This step runs on **GPU nodes**. It enhances the standardized NeMo manifest from Step 1 by adding context audio information.

Improvements over older recipes include:
- Speaker embedding extraction is run on the fly, using `torch.matmul` to compute a similarity matrix.
- It recursively finds the next best context audio if the top candidate is unsuitable, preserving more data.
- It is scaling-friendly by pre-allocating a distinct subset of speaker records to each GPU rank for balanced workloads using a greedy bin-packing strategy.
- Manifests are written out in a buffered way to reduce I/O calls.

#### Example command:
```bash
# General setup
CODE_DIR="/workspace/NeMo"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
cd ${CODE_DIR}

# Script parameters
INPUT_MANIFEST="/path/to/hifitts2/nemo_manifest/hifitts2_train.json" # From Step 1
AUDIO_BASE_DIR="/path/to/audio/files"
OUTPUT_DIR="/path/to/hifitts2/nemo_manifest"
DATASET_NAME="hifitts2" # e.g., hifitts, libritts, etc. Used for speaker ID parsing.
CONTEXT_MIN_DURATION=3.0
CONTEXT_MIN_SSIM=0.6
BATCH_SIZE=256
FLUSH_THRESHOLD_ITEMS=256
NUM_WORKERS=8
DEVICES=-1
NUM_NODES=1
WANDB_ENTITY="xyz"
WANDB_PROJECT="xyz"
WANDB_NAME="xyz"

echo "****** STEP 2: Extending NeMo Manifest with Context Audio ******"
python scripts/magpietts/extend_nemo_manifest_with_context_audio.py \
    --dataset-name ${DATASET_NAME} \
    --manifest ${INPUT_MANIFEST} \
    --audio-base-dir ${AUDIO_BASE_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --flush-threshold-items ${FLUSH_THRESHOLD_ITEMS} \
    --context-min-duration ${CONTEXT_MIN_DURATION} \
    --context-min-ssim ${CONTEXT_MIN_SSIM} \
    --batch-size ${BATCH_SIZE} \
    --devices ${DEVICES} \
    --num-nodes ${NUM_NODES} \
    --num-workers ${NUM_WORKERS} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME}
```

#### Checkout the Outputs
```bash
$ tree -L 1 hifitts2/nemo_manifest/extend_nemo_manifest_with_context_audio/
hifitts2/nemo_manifest/extend_nemo_manifest_with_context_audio/
├── hifitts2_train_rank0.json
├── hifitts2_train_rank1.json
├── hifitts2_train_rank2.json
├── hifitts2_train_rank3.json
├── hifitts2_train_rank4.json
├── hifitts2_train_rank5.json
├── hifitts2_train_rank6.json
├── hifitts2_train_rank7.json
└── hifitts2_train_withContextAudioMinDur3.0MinSSIM0.6.json   # This is the NeMo manifest used for the following steps.
```


### Step 3: Create Lhotse Shards from NeMo Manifest

This step runs on **CPU nodes**. It converts the extended NeMo manifests (from Step 2) into Lhotse shards.

Key features:
- Processes chunks of manifest entries, loads audio, and writes corresponding shard files for cuts, target audio, and context audio.
- Designed to be run in parallel worker processes.
- Loads and writes audio iteratively to save memory.

#### Example command:
```bash
# General setup
CODE_DIR="/workspace/NeMo"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
cd ${CODE_DIR}

# Script parameters
EXTENDED_MANIFEST_PATH="/path/to/hifitts2/nemo_manifest/extend_nemo_manifest_with_context_audio/hifitts2_train_withContextAudioMinDur3.0MinSSIM0.6.json" # From Step 2
AUDIO_BASE_DIR="/path/to/audio/files"
SAVE_DIR="/path/to/lhotse_shar_output"
NUM_WORKERS=16 # Number of CPU cores
SHARD_SIZE=4096

echo "****** STEP 3: Creating Lhotse Shards from NeMo Manifest ******"
python scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py \
    --manifest-path ${EXTENDED_MANIFEST_PATH} \
    --audio-base-dir ${AUDIO_BASE_DIR} \
    --output-dir ${SAVE_DIR} \
    --num-jobs ${NUM_WORKERS} \
    --processing-chunk-size ${SHARD_SIZE} \
    --audio-format 'flac' \
    --log-level 'INFO'
```

#### Checkout the outpus

```bash
$ tree -L 3 -P "*.000000.*" hifitts2/lhotse_shar/{cuts,target_audio,context_audio}
hifitts2/lhotse_shar/cuts
└── cuts.000000.jsonl.gz
hifitts2/lhotse_shar/target_audio
└── recording.000000.tar
hifitts2/lhotse_shar/context_audio
└── recording.000000.tar
```

### Step 4: Extend Lhotse Shards with Audio Codes

This final step runs on **GPU nodes**. It processes the Lhotse shards created in Step 3 to extract and add audio codes.

Improvements include:
- Pre-allocation of Lhotse shards to each rank, with each rank processing and writing independently.
- Pre-allocation of padded audio tensors, avoiding looped calls to `torch.func.pad`.
- Avoids redundant zero-padding that was present in older recipes.

#### Example command:
```bash
# General setup
CODE_DIR="/workspace/NeMo"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
cd ${CODE_DIR}

# Codec parameters
CODEC_MODEL_NAME="21fpsCausalDecoder"
CODEC_MODEL_PATH="/path/to/your/codec_model.nemo"
CODEC_FRAME_RATE=21.5

# Trainer parameters
DEVICES=-1 # Number of GPUs, -1 for all
NUM_NODES=1
BATCH_SIZE=64
WANDB_ENTITY="xyz"
WANDB_PROJECT="xyz"
WANDB_NAME="xyz"

# Path parameters
SHARD_DIR="/path/to/hifitts2/lhotse_shar" # From Step 3

echo "****** STEP 4: Extending Lhotse Shards with Audio Codes ******"
python scripts/magpietts/extend_lhotse_shards_with_audio_codes.py \
    --cuts-dir ${SHARD_DIR}/cuts \
    --target-audio-dir ${SHARD_DIR}/target_audio \
    --context-audio-dir ${SHARD_DIR}/context_audio \
    --output-dir ${SHARD_DIR} \
    --codec-model-name ${CODEC_MODEL_NAME} \
    --codec-model-path ${CODEC_MODEL_PATH} \
    --codec-frame-rate ${CODEC_FRAME_RATE} \
    --devices ${DEVICES} \
    --num-nodes ${NUM_NODES} \
    --batch-size ${BATCH_SIZE} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --log-level 'INFO' \\
```

### Checking the Outputs

After running all four steps, you can check the files by looking at the output directory specified in Steps 3 and 4.

```bash
# Examples of shard files:
$ tree -L 3 -P '*.000000.*' -I log hifitts2/lhotse_shar
hifitts2/lhotse_shar
├── codes_21fpsCausalDecoder  # This is the subdir for audio codec codes.
│   ├── context_codes
│   │   └── codes.000000.tar  # context audio codec codes.
│   └── target_codes
│       └── codes.000000.tar  # target codec codes.
├── context_audio
│   └── recording.000000.tar  # context audio waveforms.
├── cuts
│   └── cuts.000000.jsonl.gz  # Lhotse manifest.
└── target_audio
    └── recording.000000.tar  # target audio waveforms.
```

When peek one of the item from `cuts.000000.jsonl.gz`, you should expect the structure as,
```python
In [4]: cutset = CutSet.from_shar(fields={"cuts": ["hifitts2/lhotse_shar/cuts/cuts.000000.jsonl.gz"], "target_audio": ["hifitts2/lhotse_shar/target_audio/recording.000000.tar"], "context_audio": ["h
   ...: ifitts2/lhotse_shar/context_audio/recording.000000.tar"], "target_codes": ["hifitts2/lhotse_shar/codes_21fpsCausalDecoder/target_codes/codes.000000.tar"], "context_codes": ["hifitts2/lhotse_
   ...: shar/codes_21fpsCausalDecoder/context_codes/codes.000000.tar"]})

In [5]: cuts_list = [cut for cut in cutset]

In [12]: from rich import print

In [13]: print(cuts_list[0])
MonoCut(
    id='cut-rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27-0.00-3.49',
    start=0.0,
    duration=3.49,
    channel=0,
    supervisions=[
        SupervisionSegment(
            id='sup-rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27',
            recording_id='rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27',
            start=0.0,
            duration=3.49,
            channel=0,
            text='he was perhaps five years my senior',
            language='en',
            speaker='| Language:en Dataset:hifitts2 Speaker:9216 |',
            gender=None,
            custom={
                'normalized_text': 'He was perhaps five years my senior.',
                'text_source': 'mls',
                'wer': 0.0,
                'cer': 0.0,
                'speaker_count': 1,
                'bandwidth': 13953,
                'set': 'train',
                'context_speaker_similarity': 0.802838921546936,
                'context_audio_offset': 0.0,
                'context_audio_duration': 12.2,
                'context_audio_text': 'Vision of Helen," he called it, I believe.... The oblique stare of the hostile Trojans. Helen coifed with flame. Menelaus.',
                'context_audio_normalized_text': 'Vision of Helen," he called it, I believe.... The oblique stare of the hostile Trojans. Helen coifed with flame. Menelaus.',
                'context_recording_id': 'rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_30'
            },
            alignment=None
        )
    ],
    features=None,
    recording=Recording(
        id='rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27',
        sources=[AudioSource(type='file', channels=[0], source='/audio/9216/8716/9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27.flac')],
        sampling_rate=22050,
        num_samples=76955,
        duration=3.4900226757369612,
        channel_ids=[0],
        transforms=None
    ),
    custom={
        'target_audio': Recording(
            id='cut-rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_27-0.00-3.49',
            sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')],
            sampling_rate=22050,
            num_samples=76955,
            duration=3.49,
            channel_ids=[0],
            transforms=None
        ),
        'context_audio': Recording(
            id='context_cut-rec-9216-8716-9216_8716_ohenryawardstoriesof1921_1409_librivox-ohenryawardprizestoriesof1921_07_various_30-0.00-12.20',
            sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')],
            sampling_rate=22050,
            num_samples=269010,
            duration=12.2,
            channel_ids=[0],
            transforms=None
        ),
        'target_codes': TemporalArray(
            array=Array(storage_type='memory_npy', storage_path='', storage_key='<binary-data>', shape=[8, 76]),
            temporal_dim=-1,
            frame_shift=0.046511627906976744,
            start=0
        ),
        'context_codes': TemporalArray(
            array=Array(storage_type='memory_npy', storage_path='', storage_key='<binary-data>', shape=[8, 263]),
            temporal_dim=-1,
            frame_shift=0.046511627906976744,
            start=0
        ),
        'shard_origin': 'hifitts2/lhotse_shar/cuts/cuts.000000.jsonl.gz',
        'shar_epoch': 0
    }
)
```

## Extending the Existing Lhotse Shar with New Audio Codec Codes
Given existing Lhotse Shar, i.e. cuts/target_audio/context_audio, you could just run the Python script
`scripts/magpietts/extend_lhotse_shards_with_audio_codes.py` by overriding with other audio codec models. The whole
process should be the same as Step 4 as mentioned above.

## (Internal Only) Sharing Slurm Job Sub Scripts to Create Lhotse Shar 
The internal scripts for submitting these steps as Slurm jobs can be found in the GitLab repository `nemo-tts-artifacts-registry`
repository, i.e. https://gitlab-master.nvidia.com/xueyang/nemo-tts-artifacts-registry/-/tree/model_release_2505/model_release_2505/data_prep. These scripts are tailored for specific cluster environments but can be adapted for other systems.

```shell
$ tree -L 1 gitlab/nemo-tts-artifacts-registry/model_release_2505/data_prep/
gitlab/nemo-tts-artifacts-registry/model_release_2505/data_prep/
├── 1_submit_jobs_prep_input_manifest_iad.sh
├── 2_submit_jobs_extend_nemo_manifest_with_context_audio_iad.sh
├── 3_submit_jobs_create_lhotse_shards_from_nemo_manifest_iad.sh
├── 4_submit_jobs_extend_lhotse_shards_with_audio_codes_iad.sh
├── hifitts
├── hifitts2
├── jhsdGtc20Amp20Keynote
├── libritts
├── librittsDevClean
├── nvyt2505
├── README.md
├── rivaEmmaMeganSeanTom
└── rivaLindyRodney
```

