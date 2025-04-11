This guidance describes general steps on converting NeMo datasets to Lhotse Shar datasets for training
and validating Magpie-TTS. 

## Creating New Lhotse Shar Data
Step 1: reformatting `speaker` field in the NeMo manifest to pass the format check as the function defined,
```python
def check_speaker_format(item: str):
    # enforce the format as example like "| Language:en Dataset:HiFiTTS Speaker:9136_other |".
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))
```
 
Step 2: create Lhotse Shar dataset by running, 
```bash
# codec
CODEC_MODEL_NAME="21fpsCausalDecoder"
CODEC_MODEL_PATH="/codecs/21fps_causal_codecmodel.nemo"
CODEC_FRAME_RATE=21.5
SAMPLE_RATE=22050
PAD_MULTIPLE=1024

# trainer
DEVICES=-1
NUM_NODES=1
BATCH_SIZE=48
NUM_WORKERS=10
SHARD_SIZE=4096

# code
CODE_DIR="/workspace/NeMo"

# NeMo manifest
MANIFEST_PATH="/manifests/hifitts_train_withContextAudioMinDur3MinSSIM0.6.json"

# audio base dir
AUDIO_BASE_DIR="/audio/hi_fi_tts_v0"

# save dir for Shar
SAVE_DIR="/data_shar_train"

echo "*******STARTING********"
cd ${CODE_DIR}
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
echo "Starting Codec Extraction..."
python scripts/magpietts/convert_nemo_to_lhotse_shar.py
    --manifest ${MANIFEST_PATH}
    --audio_base_dir ${AUDIO_BASE_DIR}
    --save_dir ${SAVE_DIR} 
    --codec_model_name ${CODEC_MODEL_NAME}
    --codec_model_path ${CODEC_MODEL_PATH}
    --codec_frame_rate ${CODEC_FRAME_RATE}
    --sample_rate ${SAMPLE_RATE}
    --pad_multiple ${PAD_MULTIPLE}
    --devices ${DEVICES}
    --num_nodes ${NUM_NODES}
    --batch_size ${BATCH_SIZE}
    --num_workers ${NUM_WORKERS}
    --shard_size ${SHARD_SIZE}
``` 

Step 3: check the files by looking at the folder,
```shell
Examples of shard files:
$ tree data_shar_train/
 - cuts.000000.jsonl.gz  # Lhotse manifest.
 - codes_21fpsCausalDecoder.000000.tar  # target codec codes.
 - recording.000000.tar # target audio waveforms.
 - context_codes_21fpsCausalDecoder.000000.tar # context audio codec codes.
 - context_recording.000000.tar # context audio waveforms.
```

When peek one of the item from `cuts.000000.jsonl.gz`, you should expect the structure as,
```python
MonoCut(
    id='cut-audio-11614_other-12352-prideofjennico_01_castle_0000',
    start=0,
    duration=6.16,
    channel=0,
    supervisions=[
        SupervisionSegment(
            id='sup-audio-11614_other-12352-prideofjennico_01_castle_0000',
            recording_id='audio-11614_other-12352-prideofjennico_01_castle_0000',
            start=0.0,
            duration=6.16,
            channel=0,
            text='late in the year seventeen seventy one as the wind rattles the casements with impotent clutch',
            language='en',
            speaker='| Language:en Dataset:HiFiTTS Speaker:11614 |',
            gender=None,
            custom={},
            alignment=None
        )
    ],
    features=None,
    recording=Recording(
        id='audio-11614_other-12352-prideofjennico_01_castle_0000',
        sources=[
            AudioSource(
                type='memory',
                channels=[0],
                source='<binary-data>'
            )
        ],
        sampling_rate=44100,
        num_samples=271656,
        duration=6.16,
        channel_ids=[0],
        transforms=None
    ),
    custom={
        'codes_21fpsCausalDecoder': TemporalArray(
            array=Array(
                storage_type='memory_npy',
                storage_path='',
                storage_key='<binary-data>',
                shape=[8, 133]
            ),
            temporal_dim=1,
            frame_shift=0.046511627906976744,
            start=0
        ),
        'context_codes_21fpsCausalDecoder': TemporalArray(
            array=Array(
                storage_type='memory_npy',
                storage_path='',
                storage_key='<binary-data>',
                shape=[8, 138]
            ),
            temporal_dim=1,
            frame_shift=0.046511627906976744,
            start=0
        ),
        'context_recording': Recording(
            id='audio-11614_other-12220-barontrump_31_lockwood_0096',
            sources=[
                AudioSource(
                    type='memory',
                    channels=[0],
                    source='<binary-data>'
                )
            ],
            sampling_rate=44100,
            num_samples=282240,
            duration=6.4,
            channel_ids=[0],
            transforms=None
        ),
        'shard_origin': PosixPath('cuts.000000.jsonl.gz'),
        'shar_epoch': 0
    }
)
```

## Appending New Codec Codes to Existing Lhotse Manifest
TBD. In genenral, the solution is to load existing cuts of shards, attach the new codec codes to the
MonoCut's `custom` field, and write cuts and new codec codes into shard files. This should uses the 
same index of shards.

## (Internal Only) Sharing Slurm Job Sub Scripts to Create Lhotse Shar 
All scripts are stored in
https://gitlab-master.nvidia.com/xueyang/nemo-tts-artifacts-registry/-/tree/main/data_prep_lhotse .

```shell
$ tree .
.
├── extract_audioCodec_21fpsCausalDecoder_eos.sub
├── hifitts2_extract_audioCodec_21fpsCausalDecoder_eos.sub
├── README_lhotse.md
├── reserve_interactive_node.sh
└── submit_jobs_for_all_datasets.sh

$ bash submit_jobs_for_all_datasets.sh
```

