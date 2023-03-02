This is a walkthrough on how to get started on all things inpainting in this repo.

### Quick Setup

Use the machine image found [here](https://console.cloud.google.com/compute/machineImages/details/nemo-inpainter-20230221?project=citric-passage-184110)

You will almost certainly require a A100 for training this model. and at least 5000GB for holding the training data

Once you have created the instance, you will want to copy the `inpainting` folder from `/home/sam` to your home folder.

Finally you will want to re-create the virtualenv in your version of NeMo, so delete the `.env` folder in `NeMo` before running the next part


### Installating dependencies:

first create a virtualenv:

```bash
# in NeMo/
python -m venv .env
source .env/bin/activate
```

Then install the dependencies needed to run NeMo

```bash
pip install Cython
pip install -r requirements.txt --no-deps
```

There are some native libraries that are needed. Sadly I didn't record which ones those are. Any reader following these steps are encouraged to list them here after getting everything set up. If are using the GCP machine image, this should not be an issue.

### Running training (Inpainter)

To run training on the LibriTTS dataset run:
```bash
python examples/tts/inpainting.py
```

once the training is running, logs and checkpoints will be saved at `nemo_experiments/Inpainter/{date_time}`, you can use tensorboard to track the progress of the model.

#### Train on all Data

At the moment "all the data" includes libritts, vctk and VA Nicky's data.
On the machine image the data manifests are found at `data/ltts_vctk_nicky`. To train with all the data, add the arguments:
```
train_dataset=data/ltts_vctk_nicky/combined_train.json validation_datasets=data/ltts_vctk_nicky/combined_val.json sup_data_path=data/ltts_vctk_nicky/data_cache
```
to your training command

### Developement and Debugging (Inpainter)

If you want to debug or improve the training pipeline. run:

```bash
DATA_CAP=100 python examples/tts/inpainting.py
```

which will use a really small dataset of 100 examples pulled from the training set.


### Training (HiFiGan)

Training the HiFIGAN vocoder is very similar to the inpainter model. to train it on all data, run:

```bash
python examples/tts/hifigan.py train_dataset=data/ltts_vctk_nicky/combined_train.json validation_datasets=data/ltts_vctk_nicky/combined_val.json

```

### Baselines

* Inpainter model at `gs://poly-data-nl/speech/inpainter_models/inpainter_20230223`.
* Vocoder model at `gs://poly-data-nl/speech/HifiGan_models/HifiGan_20230224`


The folder structure that NeMo creates when training a model contains:

```
checkpoints/    # folder of model checkpoints
cmd-args.log    # the command arguments used to run this model
events.out.tfevents.1676891812.samml-dev-2    # tensorboard logs
git-info.log    # info on the git hash of NeMo when this model was trained
hparams.yaml    # copy of the hyperparameters used for this model

# below are log files for debugging
lightning_logs.txt
nemo_error_log.txt
nemo_log_globalrank-0_localrank-0.txt
```

# Evaluation

You can use `generate_eval_audios.py` to get reference Nickydata inpainted recordings. The script:
* (deterministically) randomly chooses a span of text to blank out
* runs the inpainter model on the audio with some part of the audio blanked out
* saves the the vocoded audio as downsampled 8k wavs (to emulate call quality audio)
* additionally saves the original audio as a downsampled wav to compare the difference

evaluation using the baseline models can be found here https://drive.google.com/drive/folders/1md2FO6_G-h70QIQekIdCmcPh3cfJ-8_-?usp=sharing

To run the valuation you need to provide:
* `inpainter_checkpoint` checkpoint to the trained inpainter model
* `vocoder_checkpoint` checkpoint to the vocoder model to use
* `dest` where to store the audio recordings
