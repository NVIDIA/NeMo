from nemo import (
    io,
    lightning as nl,
    llm,
)


class TestLoad:
    def test_reload_ckpt(self, tmpdir):
        trainer = nl.Trainer(devices=1, accelerator="cpu", strategy=nl.MegatronStrategy())
        model = llm.Mistral7BModel()
        
        ckpt = io.TrainerCheckpoint(model, trainer)
        ckpt.io_dump(tmpdir)
        loaded = io.load_ckpt(tmpdir)
        
        assert loaded.model.config.seq_length == ckpt.model.config.seq_length
