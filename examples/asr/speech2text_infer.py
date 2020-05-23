# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.****
import os
from argparse import ArgumentParser
from functools import partial

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch
from nemo.collections.asr.helpers import post_process_predictions, post_process_transcripts, word_error_rate


# Usage and Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--asr_model",
    type=str,
    default="QuartzNet15x5-En-BASE",
    required=True,
    help="Pass: 'QuartzNet15x5-En-BASE', 'QuartzNet15x5-Zh-BASE', or 'JasperNet10x5-En-Base'",
)
parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size to use for evaluation")

logging = nemo.logging
args = parser.parse_args()

def main():
    # Setup NeuralModuleFactory to control training
    # instantiate Neural Factory with supported backend
    nf = nemo.core.NeuralModuleFactory()

    # Instantiate the model which we'll train
    logging.info(f"Speech2Text: Will fine-tune from {args.asr_model}")
    asr_model = nemo_asr.models.ASRConvCTCModel.from_pretrained(model_info=args.asr_model)
    asr_model.eval()

    logging.info("\n\n")
    logging.info(f"Evaluation using {type(asr_model)} model.")
    logging.info(f"Evaluation using alphabet {asr_model.vocabulary}.")
    logging.info(f"The model has {asr_model.num_weights} weights.\n\n")

    eval_data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.dataset,
        labels=asr_model.vocabulary,
        batch_size=args.eval_batch_size,
        trim_silence=True,
        shuffle=False,
    )
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    audio_signal, audio_signal_len, transcript, transcript_len = eval_data_layer()
    log_probs, encoded_len = asr_model(input_signal=audio_signal, length=audio_signal_len)
    predictions = greedy_decoder(log_probs=log_probs)

    # inference
    eval_tensors = [log_probs, predictions, transcript, transcript_len, encoded_len]
    evaluated_tensors = nf.infer(tensors=eval_tensors)

    greedy_hypotheses = post_process_predictions(evaluated_tensors[1], asr_model.vocabulary)
    references = post_process_transcripts(evaluated_tensors[2], evaluated_tensors[3], asr_model.vocabulary)

    wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
    logging.info("Greedy WER {:.2f}%".format(wer * 100))

if __name__ == '__main__':
    main()
