# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

from tqdm.auto import tqdm

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.utils import logging
from math import ceil

def transcribe_partial_audio(
    asr_model,
    path2manifest: str,
    batch_size: int = 4,
    logprobs: bool = False,
    return_hypotheses: bool = False,
    partial_hypothesis: Optional[List['Hypothesis']] = None,
    num_workers: int = 0,
) -> List[str]:

    if isinstance(asr_model, EncDecCTCModel):
        # assert isinstance(asr_model, EncDecCTCModel), "Currently support CTC model only."

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )
        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = asr_model.training
        device = next(asr_model.parameters()).device
        dither_value = asr_model.preprocessor.featurizer.dither
        pad_to_value = asr_model.preprocessor.featurizer.pad_to

        try:
            asr_model.preprocessor.featurizer.dither = 0.0
            asr_model.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            asr_model.eval()
            # Freeze the encoder and decoder modules
            asr_model.encoder.freeze()
            asr_model.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            config = {
                'manifest_filepath': path2manifest,
                'batch_size': batch_size,
                'num_workers': num_workers,
            }

            temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
            for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                logits, logits_len, greedy_predictions = asr_model.forward(
                    input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                )
                if logprobs:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        lg = logits[idx][: logits_len[idx]]
                        hypotheses.append(lg.cpu().numpy())
                else:
                    current_hypotheses = asr_model._wer.ctc_decoder_predictions_tensor(
                        greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                    )

                    if return_hypotheses:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                    hypotheses += current_hypotheses

                del greedy_predictions
                del logits
                del test_batch

        finally:
            # set mode back to its original value
            asr_model.train(mode=mode)
            asr_model.preprocessor.featurizer.dither = dither_value
            asr_model.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                asr_model.encoder.unfreeze()
                asr_model.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses

    elif isinstance(asr_model, EncDecRNNTModel):
        # if paths2audio_files is None or len(paths2audio_files) == 0:
        #     return {}
        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []
        # Model's mode and device
        mode = asr_model.training
        device = next(asr_model.parameters()).device
        dither_value = asr_model.preprocessor.featurizer.dither
        pad_to_value = asr_model.preprocessor.featurizer.pad_to

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        try:
            asr_model.preprocessor.featurizer.dither = 0.0
            asr_model.preprocessor.featurizer.pad_to = 0

            # Switch model to evaluation mode
            asr_model.eval()
            # Freeze the encoder and decoder modules
            asr_model.encoder.freeze()
            asr_model.decoder.freeze()
            asr_model.joint.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
           
            config = {
                'manifest_filepath': path2manifest,
                'batch_size': batch_size,
                'num_workers': num_workers,
                }

            temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
            for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                encoded, encoded_len = asr_model.forward(
                    input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                )
                best_hyp, all_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_len,
                    return_hypotheses=return_hypotheses,
                    partial_hypotheses=partial_hypothesis,
                )

                hypotheses += best_hyp
                if all_hyp is not None:
                    all_hypotheses += all_hyp
                else:
                    all_hypotheses += best_hyp

                del encoded
                del test_batch
        finally:
            # set mode back to its original value
            asr_model.train(mode=mode)
            asr_model.preprocessor.featurizer.dither = dither_value
            asr_model.preprocessor.featurizer.pad_to = pad_to_value

            logging.set_verbosity(logging_level)
            if mode is True:
                asr_model.encoder.unfreeze()
                asr_model.decoder.unfreeze()
                asr_model.joint.unfreeze()
        return hypotheses, all_hypotheses


    else:
        raise ValueError("Currently support EncDecCTCModel and EncDecRNNTModel only")


    # def transcribe_partial_audio_rnnt(
    #     self,
    #     path2manifest: str,
    #     batch_size: int = 4,
    #     return_hypotheses: bool = False,
    #     partial_hypothesis: Optional[List['Hypothesis']] = None,
    #     num_workers: int = 0,
    # ) -> (List[str], Optional[List['Hypothesis']]):
    #     """
    #     Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

    #     Args:

    #         paths2audio_files: (a list) of paths to audio files. \
    #     Recommended length per file is between 5 and 25 seconds. \
    #     But it is possible to pass a few hours long file if enough GPU memory is available.
    #         batch_size: (int) batch size to use during inference. \
    #     Bigger will result in better throughput performance but would use more memory.
    #         return_hypotheses: (bool) Either return hypotheses or text
    #     With hypotheses can do some postprocessing like getting timestamp or rescoring
    #         num_workers: (int) number of workers for DataLoader

    #     Returns:
    #         A list of transcriptions in the same order as paths2audio_files. Will also return
    #     """
       