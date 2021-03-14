import contextlib
import torch

import nemo.collections.asr as nemo_asr
from nemo.utils import logging

_MODEL = None  # type: nemo_asr.models.ASRModel

# setup AMP (optional)
if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:

    @contextlib.contextmanager
    def autocast():
        yield


def is_model_availale():
    return _MODEL is not None


def get_model_names():
    model_names = set()
    for model_name in nemo_asr.models.ASRModel.list_available_models():
        if 'CTC' in model_name.class_.__name__ or 'RNNT' in model_name.class_.__name__:
            model_names.add(model_name.pretrained_model_name)
    return model_names


def initialize_model(model_name):
    global _MODEL
    if _MODEL is not None:
        del _MODEL

    # load model
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.cuda()

    # Cache model
    _MODEL = model


def transcribe_all(filepaths):
    # transcribe audio
    logging.info("Begin transcribing audio...")
    with autocast():
        with torch.no_grad():
            transcriptions = _MODEL.transcribe(filepaths, batch_size=32)
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    return transcriptions
