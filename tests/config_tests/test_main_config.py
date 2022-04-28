from omegaconf import OmegaConf


class TestConfig:

    def test_config(self):
        conf = OmegaConf.load('conf/config.yaml')
        pass
