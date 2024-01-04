import json
import sys
import tempfile
import traceback

import ipdb
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.fast_rnnt_greedy_decoding import RNNTGreedyDecodeFast
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info


def test_for_loop():
    nemo_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge", map_location="cuda")
    B = 1
    T = 100
    D = nemo_model.tokenizer.tokenizer.vocab_size  # + 1  # ?

    audio_filepath = ["/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/4446-2273-0019.wav"] * B

    conf = nemo_model.to_config_dict()
    print("GALVEZ:", json.dumps(OmegaConf.to_container(conf), indent=4))
    with open_dict(conf):
        conf["decoding"]["greedy"]["go_very_fast"] = False
        conf["decoding"]["greedy"]["max_symbols"] = 1
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        nemo_model = ASRModel.from_pretrained(
            "stt_en_fastconformer_transducer_xlarge", override_config_path=fp.name, map_location="cuda"
        )
    nemo_model.freeze()

    nemo_model.preprocessor.featurizer.dither = 0.0
    nemo_model.preprocessor.featurizer.pad_to = 0

    # Switch model to evaluation mode
    nemo_model.eval()
    # Freeze the encoder and decoder modules
    nemo_model.encoder.freeze()
    nemo_model.decoder.freeze()
    nemo_model.joint.freeze()

    actual_transcripts = nemo_model.transcribe(audio_filepath, batch_size=B)

    import ipdb

    ipdb.set_trace()

    return


if __name__ == "__main__":
    test_for_loop()
