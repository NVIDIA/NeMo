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

import copy
import os
import shutil
import tempfile

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 1024,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 1024,
        'num_classes': -1,
        'vocabulary': None,
    }

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )

    model_instance = EncDecCTCModelBPE(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModelBPE.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModelBPE)

    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
            asr_model.train()
            asr_model.save_to(save_path)

            new_model = EncDecCTCModelBPE.restore_from(save_path)
            assert isinstance(new_model, type(asr_model))
            assert new_model.vocab_path == 'vocab.txt'

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

    @pytest.mark.unit
    def test_vocab_change(self, test_data_dir, asr_model):
        old_vocab = copy.deepcopy(asr_model.decoder.vocabulary)

        with tempfile.TemporaryDirectory() as save_dir:
            save_path = os.path.join(save_dir, 'temp.nemo')

            with tempfile.TemporaryDirectory() as tmpdir:
                old_tmpdir_path = tmpdir

                old_tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
                new_tokenizer_dir = os.path.join(tmpdir, 'tokenizer')

                os.makedirs(new_tokenizer_dir, exist_ok=True)
                shutil.copy2(old_tokenizer_dir, new_tokenizer_dir)

                nw1 = asr_model.num_weights
                asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
                # No change
                assert nw1 == asr_model.num_weights

                with open(os.path.join(new_tokenizer_dir, 'vocab.txt'), 'a+') as f:
                    f.write("!\n")
                    f.write('$\n')
                    f.write('@\n')

                asr_model.change_vocabulary(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type='wpe')
                # fully connected + bias
                assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)

                new_vocab = copy.deepcopy(asr_model.decoder.vocabulary)
                assert len(old_vocab) != len(new_vocab)

                # save the model (after change of vocabulary)
                asr_model.save_to(save_path)
                assert os.path.exists(save_path)
                # delete copied version of the vocabulary from nested tmpdir (by scope)

            # assert copied vocab no longer exists
            assert not os.path.exists(os.path.join(old_tmpdir_path, 'tokenizer', 'vocab.txt'))

            # make a copy of the tokenizer before renaming
            try:
                os.rename(old_tokenizer_dir, old_tokenizer_dir + '.bkp')
                assert not os.path.exists(old_tokenizer_dir)

                # restore model from .nemo
                asr_model2 = EncDecCTCModelBPE.restore_from(save_path)
                assert isinstance(asr_model2, EncDecCTCModelBPE)

                # Check if vocabulary size is same
                assert asr_model.tokenizer.tokenizer.vocab_size == asr_model2.tokenizer.tokenizer.vocab_size

                # Make a copy of the tokenizer
                new_tokenizer_dir = os.path.join(save_dir, 'tokenizer')

                os.makedirs(new_tokenizer_dir, exist_ok=True)
                new_tokenizer_path = os.path.join(new_tokenizer_dir, 'vocab.txt')
                with open(new_tokenizer_path, 'w') as f:
                    for v in asr_model2.tokenizer.tokenizer.get_vocab():
                        f.write(f"{v}\n")

                    # Add some new tokens too
                    f.write("^\n")
                    f.write("^^\n")
                    f.write("^^^\n")

                assert os.path.exists(new_tokenizer_path)

                # change vocabulary
                asr_model2.change_vocabulary(new_tokenizer_dir, new_tokenizer_type='wpe')
                assert asr_model.tokenizer.vocab_size != asr_model2.tokenizer.vocab_size

                new_save_path = os.path.join(save_dir, 'temp2.nemo')
                asr_model2.save_to(new_save_path)

                asr_model3 = EncDecCTCModelBPE.restore_from(new_save_path)
                assert isinstance(asr_model3, EncDecCTCModelBPE)

                # Check if vocabulary size is same
                assert asr_model2.tokenizer.tokenizer.vocab_size == asr_model3.tokenizer.tokenizer.vocab_size
                assert asr_model2.tokenizer_dir != asr_model3.tokenizer_dir

                # Model PT level checks
                assert len(asr_model2.artifacts) == 1

            finally:
                os.rename(old_tokenizer_dir + '.bkp', old_tokenizer_dir)
