# Torch TTS Collection

This section of code can be used by installing the requirements inside our *requirements.txt* and *requirements_torch_tts.txt*.

## Install

This collection can be installed in the following ways:
 - pip install from github
    > ```bash
    > pip install git+https://github.com/blisc/NeMo.git@rework_reqs#egg=nemo_toolkit[torch_tts]
    > ```
  - inside a requirements file
    > `git+https://github.com/blisc/NeMo.git@rework_reqs#egg=nemo_toolkit[torch_tts]`
  - cloning from github, and then installing
    > ```bash
    >  git clone https://github.com/blisc/NeMo.git && cd NeMo && git checkout rework_reqs && pip install ".[torch_tts]"
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

from nemo.collections.tts.torch.data import TextMelAudioDataset

dataset = TextMelAudioDataset(
  manifest_filepath="PATH_TO/nvidia_ljspeech_val.json",  # Path to file that describes the location of audio and text
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
```

### Torch Modules


```python
import torch
from nemo.collections.tts.modules.hifigan_modules import Generator

gen = Generator(
    resblock = 2,
    upsample_rates = [8,8,4],
    upsample_kernel_sizes = [16,16,8],
    upsample_initial_channel = 256,
    resblock_kernel_sizes = [3,5,7],
    resblock_dilation_sizes = [[1,2], [2,6], [3,12]],
)

gen(x=torch.randn((8, 80, 100)))
```

Now you can use the hifigan generator in a pure torch setting.

## NeMo Features

If you look into the code we see that `TextMelAudioDataset` is a child of `nemo.core.classes.Dataset`. **You do not have to subclass this to add to the torch_tts repo**. It is sufficient to use `torch.utils.data.Dataset`. Using the nemo class adds some additional features not present in torch:

 - *(Optional)* Adding typing information
   - Looking at `TextMelAudioDataset`, it has a `output_types` that tells us how tensors are organized. For example, the `mels` returned by this dataset has dimensions B x D x T, which is short for saying the first dimension represents batch, the second represents a generic channels / n_mel_filters dimension, and the last represents time
 - *(Optional)* *(ToDo)* Enables serialization
   - We can now call `to_config_dict()` to return a dictionary which we can now pass to `from_config_dict()` to create another instanace of the dataset with the same arguments allowing us to easily restate code using these dictionaries. Please note to change any local paths if changing computers.

## ToDos

 - [ ] Populate *torch_tts*
   - [x] Create a new datalayer that can be used interchangeably
   - [ ] Add phone support
   - [ ] Add TTS models
 - [ ] Split Lightning away from core
   - [x] v0.1 that hacks away a lot of lightning
   - [ ] Clean up to make code less hacky
   - [ ] Split up utils better
 - [ ] Enable building *text_normlization* without installing lightning

# Appendix

## Pip install

NeMo can be installed in a number of ways. We can install just the core, collections, and even other tools such as
text normlization and inverse normalization.

How can we install these? From the NeMo folder we can run:

  - pip install {--editable} .
  - pip install {--editable} ".[all]"

In either case, we install all of the code inside the nemo folder. Even though we require librosa for speech tasks, if
we only install core, we can still run `from nemo.collections.asr import *`. It will crash saying librosa is not installed,
but the key is that the code in the asr collection is still "installed".

Is it possible to create a version of NeMo that can minimize our dependencies? Can we create collections or parts of the
code base that only relies upon pip packages that are already in our pytorch container?
This will make it easier for other teams to use standalone tools like our text normalization tools, or for other teams
to depend on only torch portions of NeMo.

My proposal is to split NeMo into two parts that can coexist:

  - Parts that depend only on torch
  - Parts that depend on ligthning, hydra, omegaconf, etc

"core"
tts -> asr, torch_tts, lightning
nlp -> lightning
asr -> lightning
all -> asr, nlp, tts

text_normalization
torch_tts

~~lightning -> core [Not an install option]~~

Make core installable. core will be our current.
New "core" will be current core minus ModelPT, exp_manager, etc.

These parts I envision are:

  - pip install .
    - The new NeMo "core"
    - Allows for simple utilities: logging, exportable
  - pip install ".[text_normalization]"
    - The text normalization tool
    - We can have other tools here as well
  - ~~pip install ".[torch_all]"~~
    - ~~A collection of our torch modules~~
  - pip install ".[torch_tts]"
    - A collection of our torch modules
  - pip install ".[~~lightning~~core]"
    - Our current "core" which includes ModelPT
  - pip install ".[~~lightning_asr~~asr]"
    - Our current asr collection. We can enable `pip install ".[asr]"` here as well for backwards compatibility
  - pip install ".[all]"
    - Installs all requirements. Everything in NeMo is expected to work here
