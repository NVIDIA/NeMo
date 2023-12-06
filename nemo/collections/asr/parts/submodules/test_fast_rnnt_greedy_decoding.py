import json
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.fast_rnnt_greedy_decoding import RNNTGreedyDecodeFast
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer

from omegaconf import open_dict
from omegaconf import OmegaConf


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

    B = 2
    T = 100
    D = nemo_model.tokenizer.tokenizer.vocab_size  # + 1  # ?

    audio_filepath = ["/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/4446-2273-0019.wav", "/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/4446-2273-0018.wav"]
    # audio_filepath = ["/home/dgalvez/scratch/data/LibriSpeech/test-clean-processed/4446-2273-0018.wav"]

    actual_transcripts = nemo_model.transcribe(audio_filepath, batch_size=B)


    # for _ in range(5):
    #     torch.zeros((1000,))
    #     torch.nn.functional.linear(torch.zeros(100, 100), torch.zeros((100,)))


    # nemo_model_fast = 
    conf = nemo_model.to_config_dict()

    print("GALVEZ:", json.dumps(OmegaConf.to_container(conf), indent=4))
    with open_dict(conf):
        conf["decoding"]["greedy"]["go_very_fast"] = True
        conf["decoding"]["greedy"]["max_symbols"] = 1
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

    # fast_transcripts = fast_model.transcribe(audio_filepath, batch_size=B)

    # fast_transcripts = fast_model.transcribe([audio_filepath[0]], batch_size=B)
    fast_transcripts = fast_model.transcribe([audio_filepath[1]], batch_size=B)
    fast_transcripts = fast_model.transcribe([audio_filepath[1]], batch_size=B)


    return

    # import ipdb; ipdb.set_trace()

    # decoding = GreedyBatchedRNNTInfer(nemo_model.decoder, nemo_model.joint, 
    #                                   blank_index=nemo_model.tokenizer.tokenizer.vocab_size, 
    #                                   max_symbols_per_step=5)
    # fast_greedy_decoder = RNNTGreedyDecodeFast(5, torch.device("cuda"), B, nemo_model.tokenizer.tokenizer.vocab_size)

    # x = torch.randn(B, T, D, dtype=torch.float32, device="cuda")
    # out_len = torch.tensor([T] * B, dtype=torch.int64, device="cuda")

    # fast_greedy_decoder(decoding, x, out_len, torch.device("cuda"))

if __name__ == "__main__":
    test_for_loop()


def test_reproducibility():
    self.preprocessor.featurizer.dither = 0.0
    self.preprocessor.featurizer.pad_to = 0

    # Switch model to evaluation mode
    self.eval()
    # Freeze the encoder and decoder modules
    self.encoder.freeze()
    self.decoder.freeze()
    self.joint.freeze()
    logging_level = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)
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
            'channel_selector': channel_selector,
        }

        if augmentor:
            config['augmentor'] = augmentor

        print("GALVEZ:augmentor=", augmentor)

        temporary_datalayer = self._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=(not verbose)):
            torch.cuda.nvtx.range_push("encoder")
            encoded, encoded_len = self.forward(
                input_signal=test_batch[0].to(device),
                input_signal_length=test_batch[1].to(device)
            )
            # print("GALVEZ:encoded=", encoded)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("decoding")
            best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=return_hypotheses,
                partial_hypotheses=partial_hypothesis,
            )
            torch.cuda.nvtx.range_pop()
    
