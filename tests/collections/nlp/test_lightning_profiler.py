import pytest
import pytorch_lightning
import pytorch_lightning.profilers
from omegaconf import DictConfig

from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder


@pytest.fixture()
def cfg():
    cfg = {
        'trainer': {
            'profiler': {
                '_target_': 'pytorch_lightning.profilers.PyTorchProfiler',
                'schedule': {
                    '_target_': 'torch.profiler.schedule',
                    'skip_first': 10,
                    'wait': 5,
                    'warmup': 1,
                    'active': 3,
                    'repeat': 2,
                },
            }
        }
    }
    return DictConfig(cfg)


class TestLightningProfiler:

    @pytest.mark.unit
    def test_pytorch_profiler(self, cfg):
        trainer = MegatronTrainerBuilder(cfg).create_trainer()
        assert isinstance(trainer.profiler, pytorch_lightning.profilers.PyTorchProfiler)
        schedule = cfg.trainer.profiler.schedule

        closure = trainer.profiler._schedule._schedule.__closure__
        captured_values = [cell.cell_contents for cell in closure]
        captured_names = trainer.profiler._schedule._schedule.__code__.co_freevars
        captured_dict = dict(zip(captured_names, captured_values))

        assert captured_dict['skip_first'] == schedule.skip_first
        assert captured_dict['wait'] == schedule.wait
        assert captured_dict['warmup'] == schedule.warmup
        assert captured_dict['active'] == schedule.active
        assert captured_dict['repeat'] == schedule.repeat
