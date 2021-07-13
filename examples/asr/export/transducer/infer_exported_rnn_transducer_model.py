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

from argparse import ArgumentParser

import hydra.utils
import torch
import tempfile
import os
import json
from tqdm import tqdm

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecRNNTModel, EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def decode_tokens_to_str(tokenizer, tokens) -> str:
    """
    Implemented by subclass in order to decoder a token list into a string.

    Args:
        tokens: List of int representing the token ids.

    Returns:
        A decoded string.
    """
    hypothesis = tokenizer.ids_to_text(tokens)
    return hypothesis


def decode_hypothesis(hypotheses_list, tokenizer, blank_id):
    """
    Decode a list of hypotheses into a list of strings.

    Args:
        hypotheses_list: List of Hypothesis.

    Returns:
        A list of strings.
    """
    for ind in range(len(hypotheses_list)):
        # Extract the integer encoded hypothesis
        prediction = hypotheses_list[ind].y_sequence

        if type(prediction) != list:
            prediction = prediction.tolist()

        # RNN-T sample level is already preprocessed by implicit CTC decoding
        # Simply remove any blank tokens
        prediction = [p for p in prediction if p != blank_id]

        # De-tokenize the integer tokens
        hypothesis = decode_tokens_to_str(tokenizer, prediction)
        hypotheses_list[ind].text = hypothesis

    return hypotheses_list


can_gpu = torch.cuda.is_available()


def main():
    encoder_model = 'Encoder-temp_rnnt.onnx'
    decoder_model = 'Decoder-Joint-temp_rnnt.onnx'
    max_symbols_per_step = 5
    decoding = ONNXGreedyBatchedRNNTInfer(encoder_model, decoder_model, max_symbols_per_step)

    nemo_model = "/home/smajumdar/PycharmProjects/nemo-eval/nemo_beta_eval/librispeech/pretrained/RNNT/Prototype/Prototype-CN-RNNT-256.nemo"
    nemo_model = EncDecRNNTBPEModel.restore_from(nemo_model)  # type: EncDecRNNTBPEModel
    tokenizer = nemo_model.tokenizer

    if torch.cuda.is_available():
        nemo_model = nemo_model.to('cuda')

    audio_filepath = ["/media/smajumdar/data/Datasets/Librispeech/LibriSpeech/test-clean-processed/61-70968-0004.wav"]

    actual_transcripts = nemo_model.transcribe(audio_filepath)[0]
    print("Actual transcripts", actual_transcripts)

    # Work in tmp directory - will store manifest file there
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for audio_file in audio_filepath:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': audio_filepath, 'batch_size': 4, 'temp_dir': tmpdir}

        temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)

        all_hypothesis = []
        for test_batch in tqdm(temporary_datalayer, desc="ONNX Transcribing"):
            input_signal, input_signal_length = test_batch[0], test_batch[1]
            if torch.cuda.is_available():
                input_signal = input_signal.to('cuda')
                input_signal_length = input_signal_length.to('cuda')

            processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            hypotheses = decoding(audio_signal=processed_audio, length=processed_audio_len)
            hypotheses = decode_hypothesis(hypotheses, tokenizer, blank_id=decoding._blank_index)  # type: List[str]

            texts = [h.text for h in hypotheses]
            all_hypothesis += hypotheses

            del processed_audio, processed_audio_len
            del test_batch

    print("ONNX transcripts", all_hypothesis)


# def main():
# parser = ArgumentParser()
# parser.add_argument(
#     "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
# )
# parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
# parser.add_argument("--batch_size", type=int, default=4)
# parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
# parser.add_argument(
#     "--dont_normalize_text",
#     default=False,
#     action='store_true',
#     help="Turn off trasnscript normalization. Recommended for non-English.",
# )
# parser.add_argument(
#     "--use_cer", default=False, action='store_true', help="Use Character Error Rate as the evaluation metric"
# )
# args = parser.parse_args()
# torch.set_grad_enabled(False)
#
# if args.asr_model.endswith('.nemo'):
#     logging.info(f"Using local ASR model from {args.asr_model}")
#     asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
# else:
#     logging.info(f"Using NGC cloud ASR model {args.asr_model}")
#     asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)
#
# asr_model.preprocessor.featurizer.pad_to = 0
# asr_model.preprocessor.featurizer.dither = 0.0
#
# asr_model.setup_test_data(
#     test_data_config={
#         'sample_rate': 16000,
#         'manifest_filepath': args.dataset,
#         'labels': asr_model.decoder.vocabulary,
#         'batch_size': args.batch_size,
#         'normalize_transcripts': not args.dont_normalize_text,
#     }
# )
# if can_gpu:
#     asr_model = asr_model.cuda()
# asr_model.eval()
# labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
# wer = WER(vocabulary=asr_model.decoder.vocabulary)
# hypotheses = []
# references = []
# for test_batch in asr_model.test_dataloader():
#     if can_gpu:
#         test_batch = [x.cuda() for x in test_batch]
#     with autocast():
#         log_probs, encoded_len, greedy_predictions = asr_model(
#             input_signal=test_batch[0], input_signal_length=test_batch[1]
#         )
#     hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
#     for batch_ind in range(greedy_predictions.shape[0]):
#         seq_len = test_batch[3][batch_ind].cpu().detach().numpy()
#         seq_ids = test_batch[2][batch_ind].cpu().detach().numpy()
#         reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
#         references.append(reference)
#     del test_batch
#
# wer_value = word_error_rate(hypotheses=hypotheses, references=references, use_cer=args.use_cer)
# if not args.use_cer:
#     if wer_value > args.wer_tolerance:
#         raise ValueError(f"got wer of {wer_value}. it was higher than {args.wer_tolerance}")
#     logging.info(f'Got WER of {wer_value}. Tolerance was {args.wer_tolerance}')
# else:
#     logging.info(f'Got CER of {wer_value}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
