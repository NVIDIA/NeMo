# Torch TTS Collection

This section of code can be used by installing the requirements inside our *requirements.txt* and *requirements_torch_tts.txt*.

## Install

This collection can be installed in the following ways:
 - pip install from github
    > ```bash
    > pip install git+https://github.com/nvidia/NeMo.git#egg=nemo_toolkit[torch_tts]
    > ```
  - inside a requirements file
    > `git+https://github.com/nvidia/NeMo.git#egg=nemo_toolkit[torch_tts]`
  - cloning from github, and then installing
    > ```bash
    >  git clone https://github.com/nvidia/NeMo.git && cd NeMo && pip install ".[torch_tts]"
    > ```

## Usage

We can check that lightning is not installed by checking pip:
```bash
pip list | grep lightning
```
Now even though lightning isn't installed, we can still use parts from the torch_tts collection.

### TTS Dataset

Let's import our dataset class and then loop through the batches. Note that in the sample .json files, we only have text
and audio. Our dataset will then create the log_mels, priors, pitches, and energies and store them in `supplementary_folder`
which in this case is `./debug0`.

```python
import torch

from nemo.collections.tts.torch.data import CharMelAudioDataset

dataset = CharMelAudioDataset(
  manifest_filepath="<PATH_TO_MANIFEST_JSON>",  # Path to file that describes the location of audio and text
  sample_rate=22050,
  supplementary_folder="./debug0",  # An additional folder that will store log_mels, priors, pitches, and energies
  max_duration=20.,  # Max duration of samples in seconds
  min_duration=0.1,  # Min duration of samples in seconds
  ignore_file=None,
  trim=False,  # Whether to use librosa.effects.trim
  n_fft=1024,
  win_length=1024,
  hop_length=256,
  window="hann",
  n_mels=64,  # Number of mel filters
  lowfreq=0,  # lowfreq for mel filters
  highfreq=8000,  # highfreq for mel filters
  pitch_fmin=80,
  pitch_fmax=640,
)

dataloader = torch.utils.data.DataLoader(dataset, 10, collate_fn=dataset._collate_fn)

for batch in dataloader:
  tokens, tokens_lengths, log_mels, log_mel_lengths, duration_priors, pitches, energies = batch
  ## Train models, etc.
  # Tokens represent already tokenized characters which probably will not work with previous tokenziers
  # You can get the label map from dataset.parser._labels_map. You can tokenize text via dataset.parser("text!")
  # You can detokenize using dataset.decode()
```

```python
import torch

from nemo.collections.tts.torch.data import PhoneMelAudioDataset

dataset = PhoneMelAudioDataset(
  manifest_filepath="<PATH_TO_MANIFEST_JSON>",  # Path to file that describes the location of audio and text
  sample_rate=22050,
  supplementary_folder="./debug0",  # An additional folder that will store log_mels, priors, pitches, and energies
)

dataloader = torch.utils.data.DataLoader(dataset, 10, collate_fn=dataset._collate_fn)

for batch in dataloader:
  tokens, tokens_lengths, log_mels, log_mel_lengths, duration_priors, pitches, energies = batch
  ## Train models, etc.
  # Tokens represent already tokenized characters which probably will not work with previous tokenziers
  # You can tokenize via dataset.vocab.encode(), and go backwards with dataset.vocab.decode().
```

## NeMo Features

If you look into the code we see that `TextMelAudioDataset` is a child of `nemo.core.classes.Dataset`. **You do not have to subclass this to add to the torch_tts repo**. It is sufficient to use `torch.utils.data.Dataset`. Using the nemo class adds some additional features not present in torch:

 - *(Optional)* Adding typing information
   - Looking at `TextMelAudioDataset`, it has a `output_types` that tells us how tensors are organized. For example, the `mels` returned by this dataset has dimensions B x D x T, which is short for saying the first dimension represents batch, the second represents a generic channels / n_mel_filters dimension, and the last represents time
 - *(Optional)* *(ToDo)* Enables serialization
   - We can now call `to_config_dict()` to return a dictionary which we can now pass to `from_config_dict()` to create another instanace of the dataset with the same arguments allowing us to easily restate code using these dictionaries. Please note to change any local paths if changing computers.

## ToDos

 - [ ] Populate *torch_tts*
   - [x] Create a new datalayer that can be used interchangeably
   - [x] Add phone support
   - [ ] Add TTS models
 - [ ] Split Lightning away from core
   - [x] v0.1 that import checks a lot of lightning
   - [ ] Split up code (core, collections, utils) better
 - [ ] Enable building *text_normlization* without installing lightning
 - [ ] Look into how `Serialization` works without hydra
