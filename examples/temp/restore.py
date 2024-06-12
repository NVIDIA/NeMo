from pathlib import Path
from pytorch_lightning import Trainer
from nemo import lightning as nl
from nemo.collections import llm


if __name__ == '__main__':

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    trainer = Trainer(  # nl.Trainer(
        devices=1, max_steps=100, accelerator='gpu', strategy=strategy, plugins=nl.MegatronMixedPrecision('bf16-mixed')
    )

    connector = None  # To demonstrate that we no longer need
    # from nemo.collections.llm.gpt.model.base_v2 import LLMSaveRestoreConnector
    # connector = LLMSaveRestoreConnector()

    # model2 = llm.GPTModelV2.restore_from('mini_gpt/', trainer=trainer, save_restore_connector=connector)
    model2 = llm.GPTModelV2.from_pretrained('smajumdar/abc5', trainer=trainer, save_restore_connector=connector)

    print("Model loaded")
    print("Num parameters: ", model2.num_weights)
