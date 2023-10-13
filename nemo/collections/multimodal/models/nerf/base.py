import pytorch_lightning as pl

from nemo.core.classes.common import Serialization
from nemo.core.classes.modelPT import ModelPT


class NerfModelBase(ModelPT, Serialization):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.save_hyperparameters()
        self._cfg = cfg

    @staticmethod
    def is_module_updatable(module):
        return hasattr(module, 'update_step') and callable(module.update_step)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass
