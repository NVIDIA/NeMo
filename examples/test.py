import hydra

from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import List

from enum import Enum
from dataclasses import dataclass, field

#@dataclass
class NeMoConfig(dict):
    """ Abstract NeMo Configuration class"""
    pass
    def serialize(self):
        """ Maybe we will have it in here - or maybe not ;) """
        pass

    def deserialize(self):
        """ Maybe we will have it in here - or maybe not ;) """
        pass

class Height(Enum):
    """ Enum - just to test how enums can be nested """
    SHORT = 0
    TALL = 1

@dataclass
class ModuleConfig(NeMoConfig):
    """ This is a module config  - just to test the inheritance """
    name: str="my_name"
    val: int=1
    height: Height = Height.SHORT # Enum!

@dataclass
class OptimConfig(NeMoConfig):
    """ This is an optimizer config """
    name: str="sgd"
    lr: float=0.99


@dataclass
class JasperBlockConfig(NeMoConfig):
    """ Default config of a single Jasper block """
    dilation: int=1
    dropout: float=0.0
    filters: int=256
    kernel: int=33
    repeat: int=5
    residual: bool=True
    separable: bool=True
    stride: int=1



@dataclass
class JasperEncoderConfig(ModuleConfig):
    """ Config of Jasper encoder """
    activation: str="relu"
    conv_mask: bool=True
    feat_in: int=64

    def __init__(self, **kwargs):
        """ Init - adds additional parameters, currently - adds 10 JasperBlocks """
        super().__init__(self, kwargs)
        # Add blocks.
        self.blocks = []
        # Add first block with some custom params.
        self.blocks.append(JasperBlockConfig(repeat=1, residual=False, stride=2))

        # Add remaining 9 using default params.
        for i in range(9):
            self.blocks.append(JasperBlockConfig())


@dataclass
class ModelConfig(NeMoConfig):
    """ This is a module config  - just to test the inheritance """
    name: str="my_name"

@dataclass
class JasperConfig(ModelConfig):
    # Encoder.
    encoder: JasperEncoderConfig = field(default_factory=JasperEncoderConfig)
    # Optimizer.
    opt: OptimConfig = field(default_factory=OptimConfig)

# Jasper "Prototype" ;)
def MyJasper(conf: JasperConfig=JasperConfig()):
    print("="*80 + " Config objects " + "="*80)
    print("Model name: ", conf.name)
    print("Model optimizer params: ", conf.opt)
    print("Model encoder activation: ", conf.encoder.activation)
    #print("Model encoder params: ", conf.encoder)
    print("Model encoder blocks: ", conf.encoder.blocks)



@hydra.main(config_path="test.yaml")
def my_app(cfg : DictConfig) -> None:
    # Pure hydra config.
    print("="*80 + " Hydra " + "="*80)
    print(cfg.pretty())

    # Jasper "object" with its default config.
    j1 = MyJasper()

    # Jasper "object" with some params overriden "in the code".
    conf2 = JasperConfig()
    conf2.name = "jasper_with_tanh"
    conf2.opt.name="adam"
    conf2.encoder.activation = "tanh"
    j2 = MyJasper(conf=conf2)


if __name__ == "__main__":
    my_app()

