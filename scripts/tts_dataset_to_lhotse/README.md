# Everything Speech Data

### Single turn speech to speech data

Our single turn speech to speech data is in the form of conversations such that it will be easy to extend to multi-turn conversations. In this section we will go through the following:

- Raw manifest format
- Lhotse cuts and shar format
- Creating Lhotse shar and cuts from raw manifest

#### Raw manifest format

Users need to get their manifests in the following format for `scripts/speech_data_generation/create_shar.py` to work. Each datapoint in the manifest should be in this format:

```
{
    'sample_id': '<UNIQUE_SAMPLE_ID | Preferably filename of source or user speech>', 
    'normalized_answer_wer': <wer of the synthesized agent speech>(Optional, this is useful if speech data was synthesized and we want to filter based on wer of the synthesized speech), 
    'normalized_answer_cer': <cer of the synthesized agent speech>(Optional, this is useful if speech data was synthesized and we want to filter based on cer of the synthesized speech), 
    'conversations': [
        {'value': '<Full Filepath of user's speech wav file>', 
        'from': 'user', 
        'type': 'audio', 
        'duration': <duration of speech wav file>(Optional | during shar creation it is automatically calculated), 
        'lang': <language of the speech>(Optional),
        'instruction': '<Assign the value "Transcribe and answer:" if the instruction is part of the speech else assign the actual value of instruction>'(This field is needed for s2s and in direct_s2s this is not needed)
        }, 
        {'value': '<Full Filepath of agent's speech wav file>', 
        'from': 'agent', 
        'type': 'audio', 
        'duration': <duration of speech wav file>(Optional | during shar creation it is automatically calculated), 
        'lang': <language of the speech>(Optional),
        'transcript': '<Transcription of the agent's speech>'
        }
    ]
}
```

#### Lhotse cuts and shar format

There will be 3 types of files generated after you run `scripts/speech_data_generation/create_shar.py`:

- cuts.{some_number}.jsonl.gz
- recording.{some_number}.tar
- target_audio.{some_number}.tar

**recording.{some_number}.tar** - tarred user (input) speech wav files

**target_audio.{some_number}.tar** - tarred agent (target) speech wav files

**cuts.{some_number}.jsonl.gz** - You can think of this as the Lhotse manifest. The format or the fields are explained as below (This document will only go over the fields which are used during training/inference)

This is what a typical cut would look like, which is one datapoint in any of the cuts.{some_number}.jsonl.gz files:
```
MonoCut(id='squadv2_5705e3a452bb891400689658-2', start=0, duration=17.345306122448978, channel=0, supervisions=[SupervisionSegment(id='squadv2_5705e3a452bb891400689658-2', recording_id='squadv2_5705e3a452bb891400689658-2', start=0, duration=17.345306122448978, channel=0, text='Transcribe and answer:', language='EN', speaker='user', gender=None, custom=None, alignment=None), SupervisionSegment(id='squadv2_5705e3a452bb891400689658-2', recording_id='squadv2_5705e3a452bb891400689658-2', start=0, duration=1.1493877551020408, channel=0, text='NCT of Delhi', language='EN', speaker='agent', gender=None, custom=None, alignment=None)], features=None, recording=Recording(id='squadv2_5705e3a452bb891400689658', sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')], sampling_rate=44100, num_samples=764928, duration=17.345306122448978, channel_ids=[0], transforms=None), custom={'target_audio': Recording(id='_lustre_fsw_portfolios_llmservice_projects_llmservice_nemo_speechlm_data_speech_QA_outputs_speechall_squadv2_train_normalized___audios_squadv2_5705e3a452bb891400689658_synthesized_normalized_answer_audio', sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')], sampling_rate=22050, num_samples=25344, duration=1.1493877551020408, channel_ids=[0], transforms=None), 'shard_origin': PosixPath('/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/s2s_lhotse_with_wavs/test2/cuts.000000.jsonl.gz'), 'shar_epoch': 0})
```

Explaination of the fields:

- id: Unique to the datapoint, this is used to reference recording wavs from the corresponding tarred files
- duration: This is the duration of the recording or source audio
- supervisions: Is a list of 2 elements (1 user turn and 1 agent turn) containing the metadata related to each turn.
- supervisions[0].text: Instruction from the user
- supervisions[0].speaker: user
- supervisions[0].language: language of input audio
- supervisions[1].text: transcript of target audio
- supervisions[1].speaker: agent
- supervisions[1].language: language of target audio
- custom['target_audio'] - This is the agent or target audio also in the form of a Recording. It has it's own duration, sampling_rate and id
- custom['target_audio'].id - is used to reference target_audios from target_audio tar file.
- custom['target_audio'].duration - self explainatory
- custom['target_audio'].sampling_rate - self explainatory

#### Creating Lhotse shar and cuts from raw manifest

To create Lhotse shar and cuts from raw manifests simple run the following command:
```
python scripts/speech_to_speech_data_generation/create_shars.py \
--manifest=<FULL PATH to raw conversation style manifest> \
--out_shar_dir=<PATH to directory where the cuts and tar files will be save> \
--num_shard=<Number of shards to create>
```