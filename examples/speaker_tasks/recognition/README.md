# Speaker Recognition

Documentation section for speaker related tasks can be found at:
 - [Speaker Identification and Verification](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)
 - [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)

## Performance
|              MODEL             |          type         | EER (%)<br>Voxceleb-O (veri_test2.txt) |
|:------------------------------:|:---------------------:|:--------------------------------------:|
| [speakerverification_speakernet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet) |        xvector        |                  1.96                  |
|           [ecapa_tdnn](https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn)           | channel-<br>attention |                  0.92                  |

## Training
Speaker Recognition models can be trained in a similar way as other models in NeMo using train and dev manifest files. Steps on how to create manifest files for voxceleb are provided below.
We provide two model configurations based on xvector and modified ecapa_tdnn, with pretrained models provided for each of them. 

For training speakernet (x-vector) model:
```bash
python speaker_reco.py --config_path='conf' --config_name='SpeakerNet_verification_3x2x256.yaml' 
```

For training ecapa_tdnn (channel-attention) model:
```bash
python speaker_reco.py --config_path='conf' --config_name='ecapa_tdnn.yaml' 
```
For step by step tutorial see [notebook](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Recognition_Verification.ipynb).

### Fine Tuning
For fine tuning on a pretrained .nemo speaker recognition model,
```bash
python speaker_reco_finetune.py --pretrained_model='/path/to/.nemo_or_.ckpt/file' --finetune_config_file='/path/to/finetune/config/yaml/file' 
```
for fine tuning tips see this [tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Recognition_Verification.ipynb)

## Inference
We provide generic scripts for manifest file creation, embedding extraction, Voxceleb evaluation and speaker ID inference. Hence most of the steps would be common and differ slightly based on your end application. 

We explain here the process for voxceleb EER calculation on voxceleb-O cleaned [trail file](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt)

### Manifest Creation
We first generate manifest file to get embeddings, The embeddings are then used by `voxceleb_eval.py` script to get EER  

```bash
# create list of files from voxceleb1 test folder (40 speaker test set)
find <path/to/voxceleb1_test/directory/> -iname '*.wav' > voxceleb1_test_files.scp
python <NeMo_root>/scripts/speaker_tasks/scp_to_manifest.py --scp voxceleb1_test_files.scp --id -3 --out voxceleb1_test_manifest.json 
```
### Embedding Extraction 
Now using the manifest file created, we can extract embeddings to `data` folder using:
```bash
python extract_speaker_embeddings.py --manifest=voxceleb1_test_manifest.json --model_path='ecapa_tdnn' --embedding_dir='./'
```
If you have a single file, you may also be using the following one liner to get embeddings for the audio file:

```python
speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="ecapa_tdnn")
embs = speaker_model.get_embedding('audio_path')
```

### Voxceleb Evaluation
``` bash
python voxceleb_eval.py --trial_file='/path/to/trail/file' --emb='./embeddings/voxceleb1_test_manifest_embeddings.pkl' 
``` 
The above command gives the performance of models on voxceleb-o cleaned trial file. 

### SpeakerID inference

To infer speaker labels on a model trained with known speaker labels (or fine tuned using pretrained model)
```bash 
python speaker_reco_infer.py --spkr_model='/path/to/.nemo/file' --train_manifest='/path/to/train/manifest/file'
--test_manifest='/path/to/test/manifest/file'
``` 

## Voxceleb Data Preparation

Scripts we provide for data preparation are very generic and can be applied to any dataset with a few path changes. 
For VoxCeleb datasets, we first download the datasets individually and make a list of audio files. Then we use the script to generate manifest files for training and validation. 
Download [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) data. 

Once downloaded and uncompressed, use programs such as ffmpeg to convert audio files from m4a format to wav format. 
Refer to the following sample command
```bash
ffmpeg -v 8 -i </path/to/m4a/file> -f wav -acodec pcm_s16le <path/to/wav/file> 
```

Generate a list file that contains paths to all the dev audio files from voxceleb1 and voxceleb2 using find command as shown below:
```bash 
find <path/to/voxceleb1/dev/folder/> -iname '*.wav' > voxceleb1_dev.scp
find <path/to/voxceleb2/dev/folder/> -iname '*.wav' > voxceleb2_dev.scp
cat voxceleb1_dev.scp voxceleb2_dev.scp > voxceleb12.scp
``` 

This list file is now used to generate training and validation manifest files using a script provided in `<NeMo_root>/scripts/speaker_tasks/`. This script has optional arguments to split the whole manifest file in to train and dev and also chunk audio files to smaller chunks for robust training (for testing, we don't need this). 

```bash
python <NeMo_root>/scripts/speaker_tasks/scp_to_manifest.py --scp voxceleb12.scp --id -3 --out voxceleb12_manifest.json --split --create_chunks
```
This creates `train.json, dev.json` in the current working directory.
