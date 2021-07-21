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

"""
This is a helper script to extract speaker embeddings based on manifest file
Usage:
python extract_speaker_embeddings.py --manifest=/path/to/manifest/file' 
--model_path='/path/to/.nemo/file'(optional)
--embedding_dir='/path/to/embedding/directory'

Args:
--manifest: path to manifest file containing audio_file paths for which embeddings need to be extracted
--model_path(optional): path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would 
    be downloaded from NGC and used to extract embeddings
--embeddings_dir(optional): path to directory where embeddings need to stored default:'./'


"""

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to manifest file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='speakerverification_speakernet',
        required=False,
        help="path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would be downloaded from NGC and used to extract embeddings",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default='./',
        required=False,
        help="path to directory where embeddings need to stored default:'./'",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.model_path.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.model_path}")
        speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(restore_path=args.model_path)
    elif args.model_path.endswith('.ckpt'):
        speaker_model = ExtractSpeakerEmbeddingsModel.load_from_checkpoint(checkpoint_path=args.model_path)
    else:
        speaker_model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name="speakerverification_speakernet")
        logging.info(f"using pretrained speaker verification model from NGC")

    num_gpus = 1 if torch.cuda.is_available() else 0
    if not num_gpus:
        logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

    trainer = pl.Trainer(gpus=num_gpus, accelerator=None)

    test_config = OmegaConf.create(
        dict(
            manifest_filepath=args.manifest,
            sample_rate=16000,
            labels=None,
            batch_size=1,
            shuffle=False,
            time_length=20,
            embedding_dir=args.embedding_dir,
        )
    )
    speaker_model.setup_test_data(test_config)
    trainer.test(speaker_model)


if __name__ == '__main__':
    main()
