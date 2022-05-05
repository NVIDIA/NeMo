# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate

from nemo.core import NeuralModule, ModelPT
from nemo.core.classes.mixins import adapter_mixins, adapter_mixin_strategies
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin, AdapterModelPTMixin


class DefaultModule(NeuralModule):
    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(50, 50)
        self.bn = torch.nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x
        return out

    def num_params(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


class DefaultModuleAdapter(DefaultModule, AdapterModuleMixin):
    def forward(self, x):
        x = super(DefaultModuleAdapter, self).forward(x)

        if self.is_adapter_available():
            # For testing purposes, cache the adapter names
            self._adapter_names = self.get_enabled_adapters()
            # call forward over model adapters, summing them up
            x = self.forward_enabled_adapters(x)

        return x


class DefaultModelAdapterMixin(AdapterModelPTMixin):
    def setup_adapters(self):
        supports_adapters = False

        # Check the inheriting class' modules supports adapters or not
        if hasattr(self, 'encoder') and isinstance(self.encoder, AdapterModuleMixin):
            supports_adapters |= True

        if hasattr(self, 'decoder') and isinstance(self.decoder, AdapterModuleMixin):
            supports_adapters |= True

        if supports_adapters:
            super().setup_adapters()

    def add_adapter(self, name: str, cfg: DictConfig):
        # setup the config for adapters
        super().add_adapter(name, cfg)

        # Try to retrieve global adapter config
        global_config = DictConfig({})
        if self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]

        # forward the method call to the individual modules
        if global_config.get('encoder_adapter', True):
            self.encoder.add_adapter(name, cfg)

        if global_config.get('decoder_adapter', False):
            self.decoder.add_adapter(name, cfg)

    def set_enabled_adapters(self, name=None, enabled: bool = True):
        # check if valid model with some adapter support
        super().set_enabled_adapters(name, enabled)

        # Try to retrieve global adapter config
        global_config = DictConfig({})
        if self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]

        # Forward the method call to the individual modules
        if global_config.get('encoder_adapter', True):
            self.encoder.set_enabled_adapters(name, enabled)

        if global_config.get('decoder_adapter', False):
            self.decoder.set_enabled_adapters(name, enabled)

    def get_enabled_adapters(self) -> list:
        enabled_adapters = super().get_enabled_adapters()

        # Try to retrieve global adapter config
        global_config = DictConfig({})
        if self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]

        # Forward the method call to the individual modules
        if global_config.get('encoder_adapter', True):
            encoder_adapters = self.encoder.get_enabled_adapters()
            enabled_adapters.extend(encoder_adapters)

        if global_config.get('decoder_adapter', True):
            decoder_adapters = self.decoder.get_enabled_adapters()
            enabled_adapters.extend(decoder_adapters)

        return enabled_adapters

    def is_adapter_available(self) -> bool:
        adapters_available = super().is_adapter_available()

        # Try to retrieve global adapter config
        global_config = DictConfig({})
        if self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]

        # Forward the method call to the individual modules
        if global_config.get('encoder_adapter', True):
            adapters_available |= self.encoder.is_adapter_available()

        if global_config.get('decoder_adapter', False):
            adapters_available |= self.decoder.is_adapter_available()

        return adapters_available

    def _check_valid_model_with_adapter_support(self):
        global_cfg = DictConfig({})
        if self.adapter_global_cfg_key in self.adapter_cfg:
            global_cfg = self.adapter_cfg[self.adapter_global_cfg_key]

        encoder_adapter = global_cfg.get('encoder_adapter', True)
        decoder_adapter = global_cfg.get('decoder_adapter', False)

        if encoder_adapter and not hasattr(self, 'encoder'):
            raise ValueError("Encoder not available")
        elif encoder_adapter and not isinstance(self.encoder, AdapterModuleMixin):
            raise ValueError("Encoder does not support adapters !")

        if decoder_adapter and not hasattr(self, 'decoder'):
            raise ValueError("Decoder is not available")
        elif decoder_adapter and not isinstance(self.decoder, AdapterModuleMixin):
            raise ValueError("Decoder does not support adapters !")


class DefaultAdapterModel(ModelPT, DefaultModelAdapterMixin):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer=trainer)

        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)

        # Required to be called for adapter support
        self.setup_adapters()

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z

    def list_available_models(cls):
        return None

    def setup_training_data(self, train_data_config):
        self._update_dataset_config('train', train_data_config)
        self._train_dl = None

    def setup_validation_data(self, val_data_config):
        self._update_dataset_config('validation', val_data_config)
        self._validation_dl = None


def get_adapter_cfg(in_features=50, dim=100, norm_pos='pre'):
    cfg = {
        '_target_': 'nemo.collections.common.parts.adapter_modules.LinearAdapter',
        'in_features': in_features,
        'dim': dim,
        'norm_position': norm_pos,
    }
    return cfg


def get_model_config(in_features=50, update_adapter_cfg: bool = True):
    config = OmegaConf.create(
        {
            'in_features': in_features,
            'encoder': {'_target_': get_classpath(DefaultModule)},
            'decoder': {'_target_': get_classpath(DefaultModule)},
        }
    )

    if update_adapter_cfg:
        enc_adapter_metadata = adapter_mixins.get_registered_adapter(config.encoder._target_)
        if enc_adapter_metadata is not None:
            config.encoder._target_ = enc_adapter_metadata.adapter_class_path

        dec_adapter_metadata = adapter_mixins.get_registered_adapter(config.decoder._target_)
        if dec_adapter_metadata is not None:
            config.decoder._target_ = dec_adapter_metadata.adapter_class_path

    return config


def update_adapter_global_cfg(cfg: DictConfig, encoder_adapter=True, decoder_adapter=False):
    if 'adapters' not in cfg:
        cfg.adapters = adapter_mixins._prepare_default_adapter_config(
            global_key=AdapterModuleMixin.adapter_global_cfg_key, meta_key=AdapterModuleMixin.adapter_metadata_cfg_key
        )

    cfg.adapters.global_cfg.encoder_adapter = encoder_adapter
    cfg.adapters.global_cfg.decoder_adapter = decoder_adapter
    return cfg


def get_classpath(cls):
    return f'{cls.__module__}.{cls.__name__}'


if adapter_mixins.get_registered_adapter(DefaultModule) is None:
    adapter_mixins.register_adapter(DefaultModule, DefaultModuleAdapter)


class TestAdapterModelMixin:
    @pytest.mark.unit
    def test_base_model_no_support_for_adapters(self):
        cfg = get_model_config(in_features=50, update_adapter_cfg=False)
        model = DefaultAdapterModel(cfg)

        with pytest.raises(ValueError):
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

    @pytest.mark.unit
    def test_single_adapter(self):
        cfg = get_model_config(in_features=50)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_all_disabled_adapters(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=False, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights

        assert new_num_params == original_num_params
        assert model.is_adapter_available() is False

    @pytest.mark.unit
    def test_enc_dec_enabled_adapters(self):
        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)
        model2 = DefaultAdapterModel(cfg)

        model2.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_encdec_num_params = model2.num_weights

        assert new_encdec_num_params > new_num_params

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multiple_adapter(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        original_num_params = new_num_params
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_linear_pre(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_linear_post(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(norm_pos='post'))
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multi_adapter_forward(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_output = model(x)

        if enc:
            assert model.encoder._adapter_names == ['adapter_0', 'adapter_1']
        if dec:
            assert model.decoder._adapter_names == ['adapter_0', 'adapter_1']

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_multi_adapter_partial_forward(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())

        model.set_enabled_adapters(name='adapter_0', enabled=False)
        new_output = model(x)

        if enc:
            assert model.encoder._adapter_names == ['adapter_1']
        if dec:
            assert model.decoder._adapter_names == ['adapter_1']
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('enc', [True, False])
    @pytest.mark.parametrize('dec', [True, False])
    def test_forward_unfrozen_adapters(self, enc, dec):
        if enc is False and dec is False:
            return  # need at least one adapter active

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=enc, decoder_adapter=dec)

        model = DefaultAdapterModel(cfg)
        original_num_params = model.num_weights

        dim = 10

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(dim=dim))
        model.freeze()
        model.unfreeze_enabled_adapters()

        assert original_num_params == 5300

        original_params = 0
        adapter_params = 0
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                assert param.requires_grad is False
                original_params += param.numel()
            else:
                assert param.requires_grad is True
                adapter_params += param.numel()

        for mname, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert module.track_running_stats is False

        assert original_params > adapter_params

        enc_params = model.encoder.num_params()
        dec_params = model.decoder.num_params()
        assert adapter_params == enc_params + dec_params

    @pytest.mark.unit
    def test_forward_linear_no_strategy(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=False)

        model = DefaultAdapterModel(cfg)
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # delete the strategy
        adapter_module = model.encoder.adapter_layer[model.get_enabled_adapters()[0]]
        del adapter_module.adapter_strategy

        with pytest.raises(AttributeError):
            _ = model(x)

    @pytest.mark.unit
    def test_forward_linear_replaced_strategy(self):
        class MultiplyAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
            def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: AdapterModuleMixin):
                out = adapter(input)
                return input * out

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        cfg = get_model_config(in_features=50)
        # Use decoder only adapter
        cfg = update_adapter_global_cfg(cfg, encoder_adapter=True, decoder_adapter=True)

        model = DefaultAdapterModel(cfg)
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # modify the strategy of both encoder and decoder
        adapter_module = model.encoder.adapter_layer[model.get_enabled_adapters()[0]]
        adapter_module.adapter_strategy = MultiplyAdapterStrategy()

        adapter_module = model.decoder.adapter_layer[model.get_enabled_adapters()[0]]
        adapter_module.adapter_strategy = MultiplyAdapterStrategy()

        out = model(x)
        # result of adapter is zero tensor, output multiplied by adapter result should be zero
        assert (out > 0.0).any() == torch.tensor(False)
