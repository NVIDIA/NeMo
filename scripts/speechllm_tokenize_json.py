import json
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf, open_dict
import torch
import os
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
import pytorch_lightning as pl
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

try:
    # main
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyNotebook as NLPDDP
except ImportError:
    # 1.21.0
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy as NLPDDP

if __name__ == '__main__':
    # Parameters
    fields_to_tokenize = [
        "answer",
        "context",
        "question"
    ]
    trainer_config = "/home/jasoli/gitrepos/NeMo/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml"
    base_config = "/home/jasoli/models/gpt_843m_gtc_tp1_pp1_1_1T/megatron_converted_843m_tp1_pp1.nemo"
    # in_json = "/mnt/drive1/data/MLS/audio_24khz_val_seen_tar_3/tarred_audio_manifest.json"
    # out_json = "/mnt/drive1/data/MLS/audio_24khz_val_seen_tar_3/tarred_audio_manifest_tokenized_256k.json"
    # in_json = "/mnt/drive1/data/MLS/audio_24khz_val_unseen_tar_3/tarred_audio_manifest.json"
    # out_json = "/mnt/drive1/data/MLS/audio_24khz_val_unseen_tar_3/tarred_audio_manifest_tokenized_256k.json"
    in_json = "/mnt/drive1/data/MLS/audio_24khz_train_tar_3/tarred_audio_manifest.json"
    out_json = "/mnt/drive1/data/MLS/audio_24khz_train_tar_3/tarred_audio_manifest_tokenized_256k.json"

    # NLP model loading setup
    config = OmegaConf.load(trainer_config)
    accelerator = 'cpu'
    config.trainer.accelerator = accelerator
    config.trainer.devices = 1
    config.trainer.precision = 32
    os.environ["LOCAL_RANK"] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'
    strategy = NLPDDP(find_unused_parameters=False, no_ddp_communication_hook=True)
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(plugins= plugins, strategy=strategy, **config.trainer)
    # Load model and get tokenizer
    model =MegatronGPTModel.restore_from(
        restore_path=base_config,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        map_location="cpu"
    )
    tokenizer = model.tokenizer
    with (
        open(in_json, "r") as in_file,
        open(out_json, "w") as out_file
    ):
        for line in in_file:
            data_i = json.loads(line)
            for field in data_i:
                if field in fields_to_tokenize:
                    data_type_key = f"{field}_type"
                    if data_type_key not in data_i.keys() or data_i[data_type_key].lower() == "text":
                        text = data_i[field]
                        tokens = tokenizer.text_to_ids(text)
                        data_i[field] = tokens
                        data_i[data_type_key] = "TOKENS"
            out_file.write(json.dumps(data_i)+'\n')
