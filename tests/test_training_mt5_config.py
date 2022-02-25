from omegaconf import OmegaConf


class TestTrainingmT5Config:
    
    def test_training_mt5_config_220m(self):
        conf = OmegaConf.load('conf/training/mt5/220m.yaml')
        s = """
        """
        expected = OmegaConf.create(s)
        # assert expected == conf, f"conf/training/mt5/220m.yaml must be set to {expected} but it currently is {conf}."


    def test_training_mt5_config_3b(self):
        conf = OmegaConf.load('conf/training/mt5/3b.yaml')
        s = """
        """
        expected = OmegaConf.create(s)
        # assert expected == conf, f"conf/training/mt5/3b.yaml must be set to {expected} but it currently is {conf}."

