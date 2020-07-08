
from omegaconf import DictConfig, MISSING
from typing import List, Any

from enum import Enum
from dataclasses import dataclass, field, asdict

from hydra import main
from hydra.core.config_store import ConfigStore

#@dataclass
class Config(object):
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
class ModuleConfig(Config):
    """ This is a module config  - just to test the inheritance """
    name: str="my_name"
    val: int=1
    height: Height = Height.SHORT # Enum!

@dataclass
class OptimConfig(Config):
    """ This is an optimizer config """
    name: str="sgd"
    lr: float=0.99


@dataclass
class JasperBlockConfig(Config):
    """ Default config of a single Jasper block """
    dilation: int=MISSING
    dropout: float=MISSING
    filters: int=MISSING
    kernel: int=MISSING
    repeat: int=MISSING
    residual: bool=MISSING
    separable: bool=MISSING
    stride: int=MISSING


@dataclass
class JasperBlock256Config(JasperBlockConfig):
    """ Default config of a single Jasper block with 256 filters """
    dilation: int=1
    dropout: float=0.0
    filters: int=256
    repeat: int=5
    residual: bool=True
    separable: bool=True
    stride=int=1

@dataclass
class JasperBlock512Config(JasperBlockConfig):
    """ Default config of a single Jasper block with 512 filters """
    dilation: int=1
    dropout: float=0.0
    filters: int=512
    repeat: int=5
    residual: bool=True
    separable: bool=True
    stride=int=1

@dataclass
class JasperBlock1024Config(JasperBlockConfig):
    """ Default config of a single Jasper block with 1024 filters """
    dilation: int=1
    dropout: float=0.0
    filters: int=1024
    kernel: 1
    repeat: int=1
    residual: bool=False
    separable: bool=True
    stride=int=1


@dataclass
class JasperEncoderConfig:
    """ Config of Jasper encoder """
    activation: str="relu"
    conv_mask: bool=True
    feat_in: int=64

    blocks: List[JasperBlockConfig] = field(default_factory=lambda: [
        # Block 1: override all values.
        JasperBlockConfig(dilation=1, dropout=0.0, filters=256, kernel=33, repeat= 1, residual=False, separable=True, stride=2),
        # Block 2-7: block 256 with different kernels.
        JasperBlockConfig(**asdict(JasperBlock256Config(kernel=33))),
        #JasperBlock256Config(kernel=33),
        #JasperBlock256Config(kernel=33),
        #JasperBlock256Config(kernel=39),
        #JasperBlock256Config(kernel=39),
        #JasperBlock256Config(kernel=39),
        # Block 8-17: block 512 with different kernels.
        #JasperBlock512Config(kernel=51),
        #JasperBlock512Config(kernel=51),
        #JasperBlock512Config(kernel=51),
        #JasperBlock512Config(kernel=63),
        #JasperBlock512Config(kernel=63),
        #JasperBlock512Config(kernel=63),
        #JasperBlock512Config(kernel=75),
        #JasperBlock512Config(kernel=75),
        #JasperBlock512Config(kernel=75),
        # Block 17: block 512 with different different dilation and no residual connection.
        #JasperBlock512Config(dilation=2, kernel=87, residual=False),
        # Last block: 1024.
        #JasperBlock1024Config()
        # This one is just to trigger an Error! with MISSING fields.
        #JasperBlockConfig()
        ])


@dataclass
class ModelConfig(Config):
    """ This is a module config  - just to test the inheritance """
    name: str="my_name"

@dataclass
class JasperConfig(ModelConfig):
    # Encoder.
    encoder: JasperEncoderConfig = JasperEncoderConfig()
    # Optimizer.
    opt: OptimConfig = OptimConfig()

@dataclass
class DatasetConfig(Config):
    name: str="imagenet"
    path: str="/datasets/imagenet"


# Jasper "Prototype" ;)
def MyJasper(conf: JasperConfig=JasperConfig()):
    print("="*80 + " Config objects " + "="*80)
    print("Model name: ", conf.name)
    print("Model optimizer params: ", conf.opt)
    print("Model encoder activation: ", conf.encoder.activation)
    #print("Model encoder params: ", conf.encoder)
    print("Model encoder blocks: ", conf.encoder.blocks)



@dataclass
class JasperAppConfig(Config):
    dataset: DatasetConfig = DatasetConfig()
    jasper: JasperConfig = JasperConfig()

cs = ConfigStore.instance()
# Registering the JasperAppConfig class with the name 'config'. 
cs.store(name="config", node=JasperAppConfig)


@main(config_name="config")
def my_app(cfg : DictConfig) -> None:
    # Pure hydra config.
    print("="*80 + " Hydra " + "="*80)
    print(cfg.pretty())

    # Jasper "object" with its default config.
    j1 = MyJasper()

    # Jasper "object" with its config from JasperAppConfig.
    j2 = MyJasper(cfg.jasper)

    # Jasper "object" with some params overriden "in the code".
    conf2 = JasperConfig()
    conf2.name = "jasper_with_tanh"
    conf2.opt.name="adam"
    conf2.encoder.activation = "tanh"
    j3 = MyJasper(conf=conf2)


if __name__ == "__main__":
    my_app()

