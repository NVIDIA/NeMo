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

import os
import tempfile

import numpy as np
import pytest
import safetensors.torch as safetensors_torch
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

try:
    from eff.cookbooks import NeMoCookbook

    _EFF_PRESENT_ = True
except ImportError:
    _EFF_PRESENT_ = False

# A decorator marking the EFF requirement.
requires_eff = pytest.mark.skipif(not _EFF_PRESENT_, reason="Export File Format library required to run test")


@pytest.fixture()
def asr_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
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
        },
    }

    decoder = {
        'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
        'params': {
            'feat_in': 1024,
            'num_classes': 28,
            'vocabulary': [
                ' ',
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z',
                "'",
            ],
        },
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


class TestFileIO:
    @pytest.mark.unit
    def test_to_from_config_file(self, asr_model):
        """" Test makes sure that the second instance created with the same configuration (BUT NOT checkpoint)
        has different weights. """

        with tempfile.NamedTemporaryFile() as fp:
            yaml_filename = fp.name
            asr_model.to_config_file(path2yaml_file=yaml_filename)
            next_instance = EncDecCTCModel.from_config_file(path2yaml_file=yaml_filename)

            assert isinstance(next_instance, EncDecCTCModel)

            assert len(next_instance.decoder.vocabulary) == 28
            assert asr_model.num_weights == next_instance.num_weights

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = next_instance.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert not np.array_equal(w1, w2)

    @pytest.mark.unit
    def test_save_restore_from_nemo_file(self, asr_model):
        """" Test makes sure that the second instance created from the same configuration AND checkpoint 
        has the same weights. """

        with tempfile.NamedTemporaryFile() as fp:
            filename = fp.name

            # Save model (with random artifact).
            with tempfile.NamedTemporaryFile() as artifact:
                asr_model.register_artifact(config_path="abc", src=artifact.name)
                asr_model.save_to(save_path=filename)

            # Restore the model.
            asr_model2 = EncDecCTCModel.restore_from(restore_path=filename)

            assert len(asr_model.decoder.vocabulary) == len(asr_model2.decoder.vocabulary)
            assert asr_model.num_weights == asr_model2.num_weights

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)

    @requires_eff
    @pytest.mark.unit
    def test_eff_save_restore_from_nemo_file_encrypted(self, asr_model):
        """" Test makes sure that after encrypted save-restore the model has the same weights. """

        with tempfile.NamedTemporaryFile() as fp:
            filename = fp.name

            # Set key - use checkpoint encryption.
            NeMoCookbook.set_encryption_key("test_key")

            # Save model (with random artifact).
            with tempfile.NamedTemporaryFile() as artifact:
                asr_model.register_artifact(config_path="abc", src=artifact.name)
                asr_model.save_to(save_path=filename)

            # Try to restore the encrypted archive (weights) without the encryption key.
            NeMoCookbook.set_encryption_key(None)
            with pytest.raises(PermissionError):
                # Restore the model.
                asr_model2 = EncDecCTCModel.restore_from(restore_path=filename)

            # Restore the model.
            NeMoCookbook.set_encryption_key("test_key")
            asr_model3 = EncDecCTCModel.restore_from(restore_path=filename)
            # Reset encryption so it won't mess up with other save/restore.
            NeMoCookbook.set_encryption_key(None)

            assert asr_model.num_weights == asr_model3.num_weights

    @pytest.mark.unit
    def test_save_restore_from_nemo_file_with_override(self, asr_model, tmpdir):
        """" Test makes sure that the second instance created from the same configuration AND checkpoint
        has the same weights.

        Args:
            tmpdir: fixture providing a temporary directory unique to the test invocation.
        """
        # Name of the archive in tmp folder.
        filename = os.path.join(tmpdir, "eff.nemo")

        # Get path where the command is executed - the artifacts will be "retrieved" there.
        # (original .nemo behavior)
        cwd = os.getcwd()

        with tempfile.NamedTemporaryFile(mode='a+') as conf_fp:

            # Create a "random artifact".
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as artifact:
                artifact.write("magic content 42")
            # Remember the filename of the artifact.
            _, artifact_filename = os.path.split(artifact.name)
            # Add artifact to model.
            asr_model.register_artifact(config_path="abc", src=artifact.name)
            # Save model (with "random artifact").
            asr_model.save_to(save_path=filename)

            # Modify config slightly
            cfg = asr_model.cfg
            cfg.encoder.activation = 'swish'
            yaml_cfg = OmegaConf.to_yaml(cfg)
            conf_fp.write(yaml_cfg)
            conf_fp.seek(0)

            # Restore the model.
            asr_model2 = EncDecCTCModel.restore_from(restore_path=filename, override_config_path=conf_fp.name)

            assert len(asr_model.decoder.vocabulary) == len(asr_model2.decoder.vocabulary)
            assert asr_model.num_weights == asr_model2.num_weights

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)

            assert asr_model2.cfg.encoder.activation == 'swish'

    @pytest.mark.unit
    def test_save_model_level_pt_ckpt(self, asr_model):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            nemo_file = os.path.join(ckpt_dir, 'asr.nemo')
            asr_model.save_to(nemo_file)

            # Save model level PT checkpoint
            asr_model.extract_state_dict_from(nemo_file, ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, asr_model._save_restore_connector.model_weights_ckpt)

            assert os.path.exists(ckpt_path)

            # Restore the model.
            asr_model2 = EncDecCTCModel.restore_from(restore_path=nemo_file)

            assert len(asr_model.decoder.vocabulary) == len(asr_model2.decoder.vocabulary)
            assert asr_model.num_weights == asr_model2.num_weights

            # Change weights values
            asr_model2.encoder.encoder[0].mconv[0].conv.weight.data += 1.0

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert not np.array_equal(w1, w2)

            # Restore from checkpoint
            asr_model2.load_state_dict(safetensors_torch.load_file(ckpt_path))

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)

    @pytest.mark.unit
    def test_save_model_level_pt_ckpt_shared_module(self, asr_model):
        class SharedModel(EncDecCTCModel):
            def __init__(self, cfg: DictConfig, trainer=None):
                super().__init__(cfg, trainer=trainer)
                self.shared = self.encoder

        shared_model = SharedModel(asr_model.cfg)
        shared_model.cfg.target = f"{SharedModel.__module__}.{SharedModel.__name__}"

        with tempfile.TemporaryDirectory() as ckpt_dir:
            nemo_file = os.path.join(ckpt_dir, 'asr.nemo')
            shared_model.save_to(nemo_file)

            # Save model level PT checkpoint
            shared_model.extract_state_dict_from(nemo_file, ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, shared_model._save_restore_connector.model_weights_ckpt)

            assert os.path.exists(ckpt_path)

            # Restore the model.
            shared_model2 = SharedModel.restore_from(restore_path=nemo_file)

            assert len(shared_model.decoder.vocabulary) == len(shared_model2.decoder.vocabulary)
            assert shared_model.num_weights == shared_model2.num_weights

            # Change weights values
            shared_model2.encoder.encoder[0].mconv[0].conv.weight.data += 1.0

            w1 = shared_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = shared_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert not np.array_equal(w1, w2)

            # Restore from checkpoint will now fail with direct safetensors.torch.load_file
            with pytest.raises(RuntimeError):
                shared_model2.load_state_dict(safetensors_torch.load_file(ckpt_path))

            # have to explicitly call the _load_state_dict_from_disk method
            shared_model2.load_state_dict(SaveRestoreConnector._load_state_dict_from_disk(ckpt_path))

            w1 = shared_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = shared_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)

    @pytest.mark.unit
    def test_save_module_level_pt_ckpt(self, asr_model):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            nemo_file = os.path.join(ckpt_dir, 'asr.nemo')
            asr_model.save_to(nemo_file)

            # Save model level PT checkpoint
            asr_model.extract_state_dict_from(nemo_file, ckpt_dir, split_by_module=True)
            encoder_path = os.path.join(ckpt_dir, 'encoder.safetensors')
            decoder_path = os.path.join(ckpt_dir, 'decoder.safetensors')
            preprocessor_path = os.path.join(ckpt_dir, 'preprocessor.safetensors')

            assert os.path.exists(encoder_path)
            assert os.path.exists(decoder_path)
            assert os.path.exists(preprocessor_path)

            # Restore the model.
            asr_model2 = EncDecCTCModel.restore_from(restore_path=nemo_file)

            assert len(asr_model.decoder.vocabulary) == len(asr_model2.decoder.vocabulary)
            assert asr_model.num_weights == asr_model2.num_weights

            # Change weights values
            asr_model2.encoder.encoder[0].mconv[0].conv.weight.data += 1.0

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert not np.array_equal(w1, w2)

            # Restore from checkpoint
            asr_model2.encoder.load_state_dict(safetensors_torch.load_file(encoder_path))

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)

    @pytest.mark.unit
    def test_save_module_level_pt_ckpt_shared_modules(self, asr_model):
        class SharedModel(EncDecCTCModel):
            def __init__(self, cfg: DictConfig, trainer=None):
                super().__init__(cfg, trainer=trainer)
                self.shared = self.encoder

        shared_model = SharedModel(asr_model.cfg)
        shared_model.cfg.target = f"{SharedModel.__module__}.{SharedModel.__name__}"

        with tempfile.TemporaryDirectory() as ckpt_dir:
            nemo_file = os.path.join(ckpt_dir, 'asr.nemo')
            shared_model.save_to(nemo_file)

            # Save model level PT checkpoint
            shared_model.extract_state_dict_from(nemo_file, ckpt_dir, split_by_module=True)
            encoder_path = os.path.join(ckpt_dir, 'encoder.safetensors')
            decoder_path = os.path.join(ckpt_dir, 'decoder.safetensors')
            preprocessor_path = os.path.join(ckpt_dir, 'preprocessor.safetensors')

            assert os.path.exists(encoder_path)
            assert os.path.exists(decoder_path)
            assert os.path.exists(preprocessor_path)

            # Restore the model.
            shared_model2 = SharedModel.restore_from(restore_path=nemo_file)

            assert len(shared_model.decoder.vocabulary) == len(shared_model2.decoder.vocabulary)
            assert shared_model.num_weights == shared_model2.num_weights

            # Change weights values
            shared_model2.encoder.encoder[0].mconv[0].conv.weight.data += 1.0

            w1 = shared_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = shared_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert not np.array_equal(w1, w2)

            # Restore from checkpoint
            shared_model2.encoder.load_state_dict(SaveRestoreConnector._load_state_dict_from_disk(encoder_path))

            w1 = shared_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = shared_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2)
