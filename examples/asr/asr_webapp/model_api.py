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


def get_model_names():
    model_names = set()
    for model_name in nemo_asr.models.ASRModel.list_available_models():
        if 'CTC' in model_name.class_.__name__ or 'RNNT' in model_name.class_.__name__:
            model_names.add(model_name.pretrained_model_name)
    return model_names


def initialize_model(model_name, use_gpu_if_available):
    # load model
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    if torch.cuda.is_available() and use_gpu_if_available:
        model = model.cuda()

    return model


def transcribe_all(filepaths, model_name, use_gpu_if_available=True):
    # instantiate model
    model = initialize_model(model_name, use_gpu_if_available)

    # transcribe audio
    logging.info("Begin transcribing audio...")
    with autocast():
        with torch.no_grad():
            transcriptions = model.transcribe(filepaths, batch_size=32)
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    # delete model
    del model
    return transcriptions
