import glob
import json
import os
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.fast_rnnt_greedy_decoding import RNNTGreedyDecodeFast
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer

from omegaconf import open_dict
from omegaconf import OmegaConf

import jiwer


import torch

import tempfile
import sys, ipdb, traceback

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()

sys.excepthook = info


def test_for_loop():
    nemo_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge",
                                          map_location="cuda")
    conf = nemo_model.to_config_dict()
    with open_dict(conf):
        conf["decoding"]["greedy"]["max_symbols"] = 1

    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        nemo_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge",
                                              override_config_path=fp.name,
                                              map_location="cuda")
    nemo_model.freeze()

    nemo_model.preprocessor.featurizer.dither = 0.0
    nemo_model.preprocessor.featurizer.pad_to = 0

    # Switch model to evaluation mode
    nemo_model.eval()
    # Freeze the encoder and decoder modules
    nemo_model.encoder.freeze()
    nemo_model.decoder.freeze()
    nemo_model.joint.freeze()

    audio_filepaths = glob.glob("/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/*.wav")[:64]
    batch_size = 16

    torch.cuda.cudart().cudaProfilerStart()

    actual_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size)

    conf = nemo_model.to_config_dict()

    with open_dict(conf):
        conf["decoding"]["greedy"]["go_very_fast"] = True
        conf["decoding"]["greedy"]["max_symbols"] = 5
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        fast_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge",
                                              override_config_path=fp.name,
                                              map_location="cuda")

    fast_model.freeze()

    fast_model.preprocessor.featurizer.dither = 0.0
    fast_model.preprocessor.featurizer.pad_to = 0

    # Switch model to evaluation mode
    fast_model.eval()
    # Freeze the encoder and decoder modules
    fast_model.encoder.freeze()
    fast_model.decoder.freeze()
    fast_model.joint.freeze()

    fast_transcripts, _ = fast_model.transcribe(audio_filepaths, batch_size=batch_size)

    print("GALVEZ:", jiwer.wer(actual_transcripts, fast_transcripts))

    for actual, fast in zip(actual_transcripts, fast_transcripts):
        if actual != fast:
            print("GALVEZ:erroneous!")
            print(actual)
            print(fast)

    torch.cuda.cudart().cudaProfilerStop()

    import ipdb; ipdb.set_trace()


def test_reproducibility():
    nemo_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge",
                                          map_location="cuda")

    conf = nemo_model.to_config_dict()
    with open_dict(conf):
        conf["decoding"]["greedy"]["go_very_fast"] = True
        conf["decoding"]["greedy"]["max_symbols"] = 5

    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        nemo_model = ASRModel.from_pretrained("stt_en_fastconformer_transducer_xlarge",
                                              override_config_path=fp.name,
                                              map_location="cuda")

    device = "cuda"

    paths2audio_files = glob.glob("/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/*.wav")
    batch_size = 16
    num_workers = 2

    nemo_model.preprocessor.featurizer.dither = 0.0
    nemo_model.preprocessor.featurizer.pad_to = 0

    # Switch model to evaluation mode
    nemo_model.eval()
    # Freeze the encoder and decoder modules
    nemo_model.encoder.freeze()
    nemo_model.decoder.freeze()
    nemo_model.joint.freeze()
    # Work in tmp directory - will store manifest file there
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in paths2audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

        config = {
            'paths2audio_files': paths2audio_files,
            'batch_size': batch_size,
            'temp_dir': tmpdir,
            'num_workers': num_workers,
            'channel_selector': None,
        }

        temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)
        for test_batch in temporary_datalayer:
            encoded, encoded_len = nemo_model.forward(
                input_signal=test_batch[0].to(device),
                input_signal_length=test_batch[1].to(device)
            )

            best_hyp, all_hyp = nemo_model.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=False,
                partial_hypotheses=None,
            )

            best_hyp_0, all_hyp_0 = nemo_model.decoding.rnnt_decoder_predictions_tensor(
                encoded[0:1, ...],
                encoded_len[0:1],
                return_hypotheses=False,
                partial_hypotheses=None,
            )

            best_hyp_1, all_hyp_1 = nemo_model.decoding.rnnt_decoder_predictions_tensor(
                encoded[1:2, ...],
                encoded_len[1:2],
                return_hypotheses=False,
                partial_hypotheses=None,
            )

            encoded_0, encoded_len_0 = nemo_model.forward(
                input_signal=test_batch[0][0:1].to(device),
                input_signal_length=test_batch[1][0:1].to(device)
            )

            best_hyp_0_single, all_hyp_0_single = nemo_model.decoding.rnnt_decoder_predictions_tensor(
                encoded_0,
                encoded_len_0,
                return_hypotheses=False,
                partial_hypotheses=None,
            )

            encoded_1, encoded_len_1 = nemo_model.forward(
                input_signal=test_batch[0][1:2].to(device),
                input_signal_length=test_batch[1][1:2].to(device)
            )

            best_hyp_1_single, all_hyp_1_single = nemo_model.decoding.rnnt_decoder_predictions_tensor(
                encoded_1,
                encoded_len_1,
                return_hypotheses=False,
                partial_hypotheses=None,
            )


            import ipdb; ipdb.set_trace()
            pass

    
if __name__ == "__main__":
    test_for_loop()
    # test_reproducibility()


