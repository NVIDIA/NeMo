# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import torch

import nemo.collections.asr as nemo_asr
from nemo.utils import logging


# setup AMP (optional)
if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

MODEL_CACHE = {}


def get_model_names():
    model_names = set()
    for model_info in nemo_asr.models.ASRModel.list_available_models():
        for superclass in model_info.class_.mro():
            if 'CTC' in superclass.__name__ or 'RNNT' in superclass.__name__:
                model_names.add(model_info.pretrained_model_name)
                logging.info(f"Available model : {model_info.pretrained_model_name}")
                break
    return model_names


def initialize_model(model_name):
    # load model
    if model_name not in MODEL_CACHE:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location='cpu')
        model.freeze()

        # cache model
        MODEL_CACHE[model_name] = model

    model = MODEL_CACHE[model_name]
    return model


def transcribe_all(filepaths, model_name, use_gpu_if_available=True):
    # instantiate model
    if model_name in MODEL_CACHE:
        model = MODEL_CACHE[model_name]
    else:
        model = initialize_model(model_name)

    if torch.cuda.is_available() and use_gpu_if_available:
        model = model.cuda()

    # transcribe audio
    logging.info("Begin transcribing audio...")
    try:
        with autocast():
            with torch.no_grad():
                transcriptions = model.transcribe(filepaths, batch_size=32)

    except RuntimeError:
        logging.info("Ran out of memory on GPU - dumping cache and performing inference on CPU for now")

        model = model.cpu()
        with torch.no_grad():
            transcriptions = model.transcribe(filepaths, batch_size=32)

    logging.info(f"Finished transcribing {len(filepaths)} files !")

    # Force onto CPU
    model = model.cpu()
    MODEL_CACHE[model_name] = model

    return transcriptions
