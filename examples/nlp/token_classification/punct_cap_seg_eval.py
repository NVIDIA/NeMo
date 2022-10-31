
import os
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf, MISSING
from pytorch_lightning import Trainer

from nemo.collections.nlp.models import PunctCapSegModel
from nemo.core.config import hydra_runner


@dataclass
class PCSEvalConfig:
    # .nemo or .ckpt file
    model: str = MISSING
    # Text file with (ideally) exactly one sentence per line. Might have to be cleaned to avoid errors. See
    # `data/prep_pcs_data.py`.
    text_file: str = MISSING
    batch_size: int = 64
    num_workers: int = 4
    # If > 1, sweep a range of thresholds and print tables. If 1, report argmax classification metrics.
    num_thresholds: int = 1
    # probably doesn't matter
    language: str = "en"
    # Set to true for Chinese, Japanese, etc.
    is_continuous: bool = False
    # Generate examples by concatenating this many lines
    min_lines_per_eg: int = 2
    max_lines_per_eg: int = 4
    # Limit examples to this many tokens
    max_length: int = 96
    # Truncate the ends up to this many
    truncate_max_tokens: int = 0
    rng_seed: int = 12345
    # Drop punctuation and apply lower-casing with these probabilities
    prob_drop_punct: float = 1.0
    prob_lower_case: float = 1.0
    # 32 or 16, to pass to PTL Trainer
    precision: int = 16


@hydra_runner(config_path="./conf/", config_name="", schema=PCSEvalConfig)
def main(cfg: PCSEvalConfig) -> None:
    m: PunctCapSegModel
    if cfg.model.endswith(".nemo"):
        if os.path.isfile(cfg.model):
            m = PunctCapSegModel.restore_from(cfg.model, map_location=torch.device("cuda"))
        else:
            m = PunctCapSegModel.from_pretrained(cfg.model, map_location=torch.device("cuda"))
    else:
        m = PunctCapSegModel.load_from_checkpoint(cfg.model, map_location="cuda")

    data_config = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "num_thresholds": cfg.num_thresholds,
        "dataset": {
            "_target_": "nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset.TextPunctCapSegDataset",
            "text_files": [cfg.text_file],
            "language": cfg.language,
            "full_stops": m.cfg.full_stops,
            "multipass": m.cfg.multipass,
            "punct_pre_labels": m.cfg.punct_pre_labels,
            "punct_post_labels": m.cfg.punct_post_labels,
            "is_continuous": cfg.is_continuous,
            "min_lines_per_eg": cfg.min_lines_per_eg,
            "max_lines_per_eg": cfg.max_lines_per_eg,
            "max_length": cfg.max_length,
            "truncate_max_tokens": cfg.truncate_max_tokens,
            "rng_seed": cfg.rng_seed,
            "prob_drop_punct": cfg.prob_drop_punct,
            "prob_lower_case": cfg.prob_lower_case,
        }
    }
    m.setup_test_data(OmegaConf.create(data_config))
    trainer = Trainer(accelerator='gpu', devices=1, precision=cfg.precision)
    result = trainer.test(m)
    print(result)


if __name__ == "__main__":
    main()
