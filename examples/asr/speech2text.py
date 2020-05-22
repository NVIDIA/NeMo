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
from nemo.collections.asr.helpers import monitor_asr_train_progress

# Usage and Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--local_rank", default=os.getenv('LOCAL_RANK', None), type=int, help="node rank for distributed training",
)
parser.add_argument(
    "--amp_opt_level",
    default="O0",
    type=str,
    choices=["O0", "O1", "O2", "O3"],
    help="See: https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--asr_model",
    type=str,
    default="QuartzNet15x5-En-BASE",
    required=True,
    help="Pass: 'QuartzNet15x5-En-BASE', 'QuartzNet15x5-Zh-BASE', or 'JasperNet10x5-En-Base' to train from pre-trained models. To train from scratch pass path to modelfile ending with .yaml.",
)
parser.add_argument(
    "--train_dataset", type=str, required=True, default=None, help="training dataset path",
)
parser.add_argument("--batch_size", required=True, type=int, help="train batch size per GPU")


args = parser.parse_args()

logging = nemo.logging


def main():
    # Setup NeuralModuleFactory to control training
    # instantiate Neural Factory with supported backend
    nf = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,  # This is necessary for distributed training
        optimization_level=args.amp_opt_level,  # This is necessary for mixed precision optimization
        cudnn_benchmark=True,
    )
    logging.info(f"Speech2Text: Training on {nf.world_size} GPUs.")

    # Instantiate the model which we'll train
    if args.asr_model.endswith('.yaml'):
        logging.info(f"Speech2Text: Will train from scratch using config from {args.asr_model}")
        asr_model = nemo_asr.models.ASRConvCTCModel.import_from_config(args.asr_model)
    else:
        logging.info(f"Speech2Text: Will fine-tune from {args.asr_model}")
        asr_model = nemo_asr.models.ASRConvCTCModel.from_pretrained(
            model_info=args.asr_model, local_rank=args.local_rank
        )

    train_data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        labels=asr_model.vocabulary,
        batch_size=args.batch_size,
        trim_silence=True,
        max_duration=16.7,
        shuffle=True,
    )
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(asr_model.vocabulary))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    audio_signal, audio_signal_len, transcript, transcript_len = train_data_layer()
    log_probs, encoded_len = asr_model(input_signal=audio_signal, length=audio_signal_len)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len)

    # START TRAINING
    tensors_to_evaluate = [predictions, transcript, transcript_len]
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss] + tensors_to_evaluate,
        print_func=partial(monitor_asr_train_progress, labels=asr_model.vocabulary),
    )
    nf.train(
        tensors_to_optimize=[loss],
        callbacks=[train_callback],
        optimizer="novograd",
        optimization_params={"num_epochs": 30, "lr": 0.001, "weight_decay": 1e-3},
    )
    # Export trained model


if __name__ == '__main__':
    main()
