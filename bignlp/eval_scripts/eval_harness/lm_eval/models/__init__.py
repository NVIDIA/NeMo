from lm_eval.base import LM
from . import nemo_gpt3, nemo_gpt3_tp_pp
from . import dummy

MODEL_REGISTRY = {
    "nemo-gpt3-tp-pp": nemo_gpt3_tp_pp.NeMo_GPT3LM_TP_PP,
    "nemo-gpt3": nemo_gpt3.NeMo_GPT3LM_TP_PP,
    "dummy": dummy.DummyLM,
}


def get_model(model_name: str) -> LM:
    return MODEL_REGISTRY[model_name]
