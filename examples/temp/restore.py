from pathlib import Path
from pytorch_lightning import Trainer
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.base_v2 import LLMSaveRestoreConnector


if __name__ == '__main__':

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    trainer = Trainer(  # nl.Trainer(
        devices=1,
        max_steps=100,
        accelerator='gpu',
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision('bf16-mixed')
    )

    # temp hack
    connector = LLMSaveRestoreConnector()
    # connector.model_extracted_dir = 'mini_gpt/' 

    # model2 = llm.GPTModelV2.restore_from('mini_gpt/', trainer=trainer, save_restore_connector=connector)
    model2 = llm.GPTModelV2.from_pretrained('smajumdar/abc5', trainer=trainer, save_restore_connector=connector)
    print(list(model2.state_dict().keys()))
    
    # keys = model.trainer.strategy.megatron_parallel.sharded_state_dict().keys()
    # sd1 = model.trainer.strategy.megatron_parallel.sharded_state_dict()
    # sd2 = model2.trainer.strategy.megatron_parallel.sharded_state_dict()
    # for key in keys:
    #     v1 = sd1[key]
    #     v2 = sd2[key]
    #     assert (v1 - v2).abs().mean() < 1e-5