from lm_eval.base import LM
from . import gpt2
from . import gpt3
from . import nemo_gpt3
from . import nemo_gpt3_tp
from . import turing_api
from . import dummy

MODEL_REGISTRY = {
    # "gpt2": gpt2.GPT2LM,
    # "gpt3": gpt3.GPT3LM,
    "nemo-gpt3": nemo_gpt3.NeMo_GPT3LM,
    "nemo-gpt3-tp": nemo_gpt3_tp.NeMo_GPT3LM_TP,
    # "turing-api": turing_api.ApiLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name: str) -> LM:
    return MODEL_REGISTRY[model_name]
