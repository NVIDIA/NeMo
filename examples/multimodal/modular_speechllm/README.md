# Modular SpeechLLM

This directory contains example scripts to train and evaluate modular SpeechLLM models [1]. 

## Requirements
You will need to install this specific branch of NeMo, or use the provided Dockerfile in the root directory of this repository to build a Docker image with all the necessary dependencies. This branch is based on NeMo main branch by 2/14/2024, while diverging from the main branch in the following ways:
- Migrating to pytorch_lightning==2.2 to fix some bugs with multiple validation dataloader_iter and saving -last.ckpt files.
- Pinning to megatron-core==0.4.0 to avoid possible unstable behavior of the latest versions or not well supported NeMo components.


## Architecture

In general, there're three main components of a modular SpeechLLM model: 
- An audio encoder that processes the input audio and produces a sequence of audio embeddings.
- A modality adapter that processes the audio embeddings and produces a sequence of embeddings in the same latent space as the token embeddings of a pretrained large language model (LLM).
- A pretrained large language model (LLM) that processes embeddings from the modality adapter as well as token embeddings of input prompt, and produces the text output. The audio embeddings and text token embeddings are concatenated in time dimension before going into the LLM.


## Usage

### Input Format

You'll need to prepare data in the NeMo manifest format, where each line is a python dictionary with some keys, for example:
```
{
    "audio_filepath": "path/to/audio.wav",
    "offset": 0.0, # offset of the audio in seconds, this is an optional field
    "duration": 10.0 , # duration of the audio in seconds, can set to `None` to load the whole audio
    "question": "what is the transcription of the audio?", this is an optional field, see below for more details
    "answer": "the transcription of the audio",
}
```

The `question` field in the manifest is optional, and you can put a list of questions in a file then set `++model.data.train_ds.question_file=<path to to question file>` to ask the dataloader to randomly pick a question from the file for each audio sample. This is useful for training with multiple prompts for the same task. If neither `question` field nor `question_file` is provided, the dataloader will use a default question `what does the audio mean?` for all aduios.


### Training

There are several configs for training a SpeechLLM:
- `conf/modular_audio_gpt_config_peft.yaml`: a config for training a SpeechLLM model with PEFT (e.g., LoRA), where you don't want to tune the whole LLM but still want to adapt the LLM to your needs.
- `conf/modular_audio_gpt_config_sft.yaml`: a config for training a SpeechLLM model without PEFT, where you might want to tune the whole LLM or simply freeze it and use as is.
- `conf/modular_audio_gpt_multi_enc_config_peft.yaml`: a config for training a SpeechLLM model with multiple audio encoders and PEFT, where you can add speaker embeddings to the audio embeddings. Currently only TitaNet is supported as the speaker encoder.

With any config, you can set the following flags to control which components to train or freeze:
- `model.freeze_llm`: Generally set to `True` unless you want to fine-tune the whole LLM.
- `model.freeze_audio_encoder`: Generally set to `False` unless you want to freeze the audio encoder.
- `model.freeze_modality_adapter`: Generally set to `False` since we want to train the modality adapter.

In addition to the config file, you will also need two prepare the audio encoder and the LLM as `*.nemo` files.

To train a SpeechLLM model, you can run the following script:
```bash
MEGATRON_MODEL=/path/to/megatron-model.nemo
ASR_MODEL=/path/to/audio-encoder.nemo

TRAIN_MANIFESTS="[/data/train_1.json,/data/train_2.json]"
VAL_MANIFESTS="[/data/dev_1.json,/data/dev_2.json]"
VAL_NAMES="[dev-1,dev-2]"

NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
NVTE_MASKED_SOFTMAX_FUSION=0 \
CUDA_VISIBLE_DEVICES="0,1" python modular_audio_gpt_train.py --config-path="./conf" --config-name "modular_audio_gpt_config_peft" \
    trainer.devices=-1 \
    model.freeze_audio_encoder=True \
    model.freeze_llm=True \
    model.global_batch_size=4 \  # global_batch_size = micro_batch_size * num_gpus_per_node * num_nodes * gradient_accumulation_steps
    model.micro_batch_size=2 \  # micro_batch_size = batch_size_per_gpu
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_MODEL \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.validation_ds.names=$VAL_NAMES \
```

You can also use tarred datasets for faster training by converting normal NeMo datasets to tarred datasets using this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/convert_to_tarred_audio_dataset.py) and follow the same dataset setting as shown in the script.


#### Multi-task training
In order to use a question file, you can set `++model.data.train_ds.question_file=<path to to question file>` in the command line or use multiple question files with `++model.data.train_ds.question_file=[<path to to question file1>,<path to question file2>,...]`. If the number of question files is equal to the number of provided datasets, the dataloader will assigne each question file to a dataset. Otherwise, the dataloader will randomly pick a question file from all provided question files for each audio sample. Using multiple question files is useful for training with multiple tasks, where each task has its own set of prompts. Meanwhile, you can control the weights for different tasks/datasets by using concatentated tarred datasets, where you can assign weights to datasets by:
```
++model.data.train_ds.is_tarred=True \
++model.data.train_ds.is_concat=True \
++model.data.train_ds.manifest_filepath=[/path/to/data1/tarred_audio_manifest.json,/path/to/data2/tarred_audio_manifest.json] \
++model.data.train_ds.tarred_audio_filepaths=[/path/to/data1/audio__OP_0..1023_CL_.tar,/path/to/data2/audio__OP_0..1023_CL_.tar] \
++model.data.train_ds.concat_sampling_technique='random' \
++model.data.train_ds.concat_sampling_probabilities=[0.4,0.6] \
```

#### Available Audio Encoders
Currently all NeMo ASR models are supported, others may also work if they have an `encoder` attribute that returns a sequence of audio embeddings, and a `preprocessor` that takes raw audios and returns a sequence of features for the encoder. The model should also have a `cfg` attribute that returns a `omegaconf.DictConfig` object of model configuration. In addition to a local model, you can also set `pretrained_audio_model` to a model from NGC (e.g., `stt_en_fastconformer_transducer_large`) or Huggingface (e.g., `nvidia/parakeet-rnnt-1.1b`), and the script will download the model and use it for training.


### Inference

The config file for inference is `conf/modular_audio_gpt_config_eval.yaml`, where you mainly need to set the `model.data.test_ds` fields. An example of running inference is shown below:

```bash
ASR_MODEL=/path/to/asr-model.nemo  # required only if you freeze the audio encoder during training
MEGATRON_CKPT=/path/to/megatron-llm.nemo
ALM_DIR=/path/to/nemo_experiments/job_name
ALM_YAML=$ALM_DIR/version_0/hparams.yaml
ALM_CKPT="$ALM_DIR/checkpoints/AudioGPT--validation_wer\=0.2-step\=100000-epoch\=0-last.ckpt"  # this checkpoint file only contains the trainable params

VAL_MANIFESTS="[/data/libri-test-other.json,/data/MCV_7.1_test.json,/data/wsj-test.json]"
VAL_NAMES="[ls-test-other,mcv7.1-test,wsj-test]"

NVTE_MASKED_SOFTMAX_FUSION=0 \
NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
CUDA_VISIBLE_DEVICES=0 python modular_audio_gpt_eval.py \
    model.restore_from_path=$MEGATRON_CKPT \
    model.pretrained_audio_model=$ASR_MODEL \  # required only if you freeze the audio encoder during training
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.test_ds.names=$VAL_NAMES \
    model.data.test_ds.global_batch_size=8 \
	model.data.test_ds.micro_batch_size=8 \
	model.data.test_ds.tokens_to_generate=256 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.output_dir=${ALM_DIR}
```


## Reference
[1] Chen, Z.\*, Huang, H.\*, Andrusenko, A., Hrinchuk, O., Puvvada, K.C., Li, J., Ghosh, S., Balam, J. and Ginsburg, B., 2023. SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation. ICASSP'24.
