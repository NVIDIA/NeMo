# Speaker Dirarzation

Documentation section for speaker related tasks can be found at:
 - [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)
 - [Speaker Identification and Verification](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)


## Features of NeMo Speaker Diarization
- Provides pretrained speaker embedding extractor models and VAD models.
- Does not need to be tuned on dev-set while showing the better performance than AHC+PLDA method in general.
- Estimates the number of speakers in the given session.
- Provides example script for asr transcription with speaker labels. 


## Performance
Diarization Error Rate (DER) table of `ecapa_tdnn.nemo` model on well known evaluation datasets. 

|         Evaluation<br>Condition     | NIST SRE 2000 | AMI<br>(Lapel) | AMI<br>(MixHeadset) | CH109 |
|:-----------------------------------:|:-------------:|:--------------:|:-------------------:|:-----:|
|  Oracle VAD <br>KNOWN # of Speakers  |      7.1     |      1.94      |         2.31        |  1.19 |
| Oracle VAD<br> UNKNOWN # of Speakers |     6.78     |      2.58      |         2.13        |  1.73 |

* All models were tested using embedding extractor with window size 1.5s and shift length 0.75s
* The above result is based on the oracle Voice Activity Detection (VAD) result.
* This result is based on [ecapa_tdnn.nemo](https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn) model which will be soon uploaded on NGC.

<br/>

## Run Speaker Diarization on Your Audio Files

#### Example script
```bash
python speaker_diarize.py \
    diarizer.paths2audio_files='my_wav.list' \
    diarizer.speaker_embeddings.model_path='speakerdiarization_speakernet' \
    diarizer.oracle_num_speakers=null \
    diarizer.vad.model_path='vad_telephony_marblenet' 
```

If you have oracle VAD files and groundtruth RTTM files for evaluation:

```bash
python speaker_diarize.py \
    diarizer.paths2audio_files='my_wav.list' \
    diarizer.speaker_embeddings.model_path='path/to/speakerdiarization_speakernet.nemo' \
    diarizer.oracle_num_speakers= null \
    diarizer.speaker_embeddings.oracle_vad_manifest='oracle_vad.manifest' \
    diarizer.path2groundtruth_rttm_files='my_wav_rttm.list' 
```

#### Arguments
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To run speaker diarization on your audio recordings, you need to prepare the following files.

- **`diarizer.paths2audio_files`: audio file list**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Provide a list of full path names to the audio files you want to diarize.

Example: `my_wav.list`

```bash
/path/to/my_audio1.wav
/path/to/my_audio2.wav
/path/to/my_audio3.wav
```

- **`diarizer.oracle_num_speakers` : number of speakers in the audio recording**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If you know how many speakers are in the recordings, you can provide the number of speakers as follows.

Example: `number_of_speakers.list`
```
my_audio1 7
my_audio2 6
my_audio3 5
<session_name> <number of speakers>
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.oracle_num_speakers='number_of_speakers.list'` 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If all sessions have the same number of speakers, you can specify the option as: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.oracle_num_speakers=5`

- **`diarizer.speaker_embeddings.model_path`: speaker embedding model name**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Specify the name of speaker embedding model, then the script will download the model from NGC. Currently, we have 'speakerdiarization_speakernet' and 'speakerverification_speakernet'.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.speaker_embeddings.model_path='speakerdiarization_speakernet'`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You could also download *.nemo files from [this link](https://ngc.nvidia.com/catalog/models?orderBy=scoreDESC&pageNumber=0&query=SpeakerNet&quickFilter=&filters=) and specify the full path name to the speaker embedding model file (`*.nemo`).

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.speaker_embeddings.model_path='path/to/speakerdiarization_speakernet.nemo'` 
 
- **`diarizer.vad.model_path`: voice activity detection modle name or path to the model**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Specify the name of VAD model, then the script will download the model from NGC. Currently, we have 'vad_marblenet' and  'vad_telephony_marblenet' as options for VAD models.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.vad.model_path='vad_telephony_marblenet'`


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Instead, you can also download the model from [vad_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_marblenet) and [vad_telephony_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_telephony_marblenet) and specify the full path name to the model as below.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.vad.model_path='path/to/vad_telephony_marblenet.nemo'`


- **`diarizer.path2groundtruth_rttm_files`: reference RTTM files for evaluating the diarization output**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can provide a list of rttm files for evaluating the diarization output.

Example: `my_wav_rttm.list`
```
/path/to/my_audio1.rttm
/path/to/my_audio2.rttm
/path/to/my_audio3.rttm
<full path to *.rttm file>
```

<br/>

## Run Speech Recognition with Speaker Diarization

Using the script `asr_with_diarization.py`, you can transcribe your audio recording with speaker labels as shown below:

```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going 
[00:04.46 - 00:09.96] speaker_1: oh pretty well it was really crowded today yeah i kind of assumed everylonewould be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
```

Currently, asr_with_diarization only supports QuartzNet English model ([`QuartzNet15x5Base`](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#id110)). 

#### Example script

```bash
python asr_with_diarization.py \
    --pretrained_speaker_model='speakerdiarization_speakernet' \
    --audiofile_list_path='my_wav.list' \
```
If you have reference rttm files or oracle number of speaker information, you can provide those files as in the following example.

```bash
python asr_with_diarization.py \
    --pretrained_speaker_model='speakerdiarization_speakernet' \
    --audiofile_list_path='my_wav.list' \
    --reference_rttmfile_list_path='my_wav_rttm.list'\
    --oracle_num_speakers=number_of_speakers.list
```

#### Output folders

The above script will create a folder named `./asr_with_diar/`.
In `./asr_with_diar/`, you can check the results as below.

```bash
./asr_with_diar
├── json_result
│   └── my_audio1.json
│   └── my_audio2.json
│   └── my_audio3.json
│
└── transcript_with_speaker_labels
    └── my_audio1.txt
    └── my_audio2.txt
    └── my_audio3.txt
│
└── ...
```


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `json_result` folder includes word-by-word json output with speaker label and time stamps.

Example: `./asr_with_diar/json_result/my_audio1.json`
```bash
{
    "status": "Success",
    "session_id": "my_audio1",
    "transcription": "back from the gym oh good ...",
    "speaker_count": 2,
    "words": [
        {
            "word": "back",
            "start_time": 0.44,
            "end_time": 0.56,
            "speaker_label": "speaker_0"
        },
...
        {
            "word": "oh",
            "start_time": 1.74,
            "end_time": 1.88,
            "speaker_label": "speaker_1"
        },
        {
            "word": "good",
            "start_time": 2.08,
            "end_time": 3.28,
            "speaker_label": "speaker_1"
        },
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `transcript_with_speaker_labels` folder includes transcription with speaker labels and corresponding time.

Example: `./asr_with_diar/transcript_with_speaker_labels/my_audio1.txt`
```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going
[00:04.46 - 00:09.96] speaker_1: pretty well it was really crowded today yeah i kind of assumed everylonewould be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
[00:13.97 - 00:15.78] speaker_1: but it's the fourth of july mm
[00:16.90 - 00:21.80] speaker_0: so yeahg people still work tomorrow do you have to work tomorrow did you drive off yesterday
```

