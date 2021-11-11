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
* This result is based on [ecapa_tdnn.nemo](https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn) model.

<br/>

## Run Speaker Diarization on Your Audio Files

#### Example script
```bash
  python offline_diarization.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.vad.model_path=<pretrained modelname or path to .nemo>
```

If you have oracle VAD files and groundtruth RTTM files for evaluation:
Provide rttm files in the input manifest file and enable oracle_vad as shown below. 

```bash
python offline_diarization.py \
  python speaker_diarize.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.oracle_vad=True
```

#### Arguments
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To run speaker diarization on your audio recordings, you need to prepare the following file.

- **`diarizer.manifest_filepath`: <manifest file>** Path to manifest file 

Example: `manifest.json`

```bash
{"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, label: "infer", "text": "-", "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}
```
Mandatory fields are `audio_filepath`, `offset`, `duration`, `label:"infer"` and `text: <ground truth or "-" >`  , and the rest are optional keys which can be passed based on the type of evaluation 

Some of important options in config file: 

- **`diarizer.speaker_embeddings.model_path`: speaker embedding model name**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Specify the name of speaker embedding model, then the script will download the model from NGC. Currently, we have 'ecapa_tdnn' and 'speakerverification_speakernet'.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.speaker_embeddings.model_path='ecapa_tdnn'`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You could also download *.nemo files from [this link](https://ngc.nvidia.com/catalog/models?orderBy=scoreDESC&pageNumber=0&query=SpeakerNet&quickFilter=&filters=) and specify the full path name to the speaker embedding model file (`*.nemo`).

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.speaker_embeddings.model_path='path/to/ecapa_tdnn.nemo'` 
 
- **`diarizer.vad.model_path`: voice activity detection modle name or path to the model**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Specify the name of VAD model, then the script will download the model from NGC. Currently, we have 'vad_marblenet' and  'vad_telephony_marblenet' as options for VAD models.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.vad.model_path='vad_telephony_marblenet'`


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Instead, you can also download the model from [vad_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_marblenet) and [vad_telephony_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_telephony_marblenet) and specify the full path name to the model as below.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `diarizer.vad.model_path='path/to/vad_telephony_marblenet.nemo'`

<br/>

## Run Speech Recognition with Speaker Diarization

Using the script `offline_diarization_with_asr.py`, you can transcribe your audio recording with speaker labels as shown below:

```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going 
[00:04.46 - 00:09.96] speaker_1: oh pretty well it was really crowded today yeah i kind of assumed everylonewould be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
```

Currently, offline_diarization_with_asr supports QuartzNet English model and ConformerCTC model (`QuartzNet15x5Base-En`, `stt_en_conformer_ctc_large`). 

#### Example script

```bash
python offline_diarization_with_asr.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.parameters.asr_based_vad=True
```
If you have reference rttm files or oracle number of speaker information, you can provide those file paths and number of speakers in the manifest file path and pass `diarizer.clustering.parameters.oracle_num_speakers=True` as shown in the following example.

```bash
python offline_diarization_with_asr.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.parameters.asr_based_vad=True \
    diarizer.clustering.parameters.oracle_num_speakers=True
```

#### Output folders

The above script will create a folder named `./demo_asr_output/`.
In `./demo_asr_output/`, you can check the results as below.

```bash
./asr_with_diar
├── pred_rttms
    └── my_audio1.json
    └── my_audio1.txt
    └── my_audio1.rttm
│
└── ...
```


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `*.json` files contains word-by-word json output with speaker label and time stamps.

Example: `./demo_asr_output/pred_rttms/my_audio1.json`
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `*.txt` files contain transcription with speaker labels and corresponding time.

Example: `./demo_asr_output/pred_rttms/my_audio1.txt`
```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going
[00:04.46 - 00:09.96] speaker_1: pretty well it was really crowded today yeah i kind of assumed everylonewould be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
[00:13.97 - 00:15.78] speaker_1: but it's the fourth of july mm
[00:16.90 - 00:21.80] speaker_0: so yeahg people still work tomorrow do you have to work tomorrow did you drive off yesterday
```
