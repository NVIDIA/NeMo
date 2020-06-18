
from .jarvis import asr_engine

def backends():
    return ["jarvis-jasper"]

def backends_str():
    return ' | '.join(backends())

def create_engine(backend="jarvis-jasper", **kwargs):
    if backend not in backends():
        raise ValueError("invalid ASR backend:  {:s}  (valid ASR backends are:  {:s})".format(backend, backends_str()))

    print('ASR - creating ASR backend:  ' + backend)

    if backend == "jarvis-jasper":
        return asr_engine.JarvisAsrEngine(**kwargs)

