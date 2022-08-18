import sys

import copy
import math
import hydra
import omegaconf
import subprocess

from bignlp.core.stages import BigNLPStage
from bignlp.core.stages import Training, FineTuning, PromptLearning
from bignlp.core.stages import Conversion
from bignlp.core.stages import EvalHarnessEvaluation, NeMoEvaluation
from bignlp.core.data_stages import PileDataPreparation, MC4DataPreparation, CustomDataPreparation


omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
omegaconf.OmegaConf.register_new_resolver("divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True)
omegaconf.OmegaConf.register_new_resolver("divide_floor", lambda x, y: int(math.floor(x / y)), replace=True)

STR2STAGECLASS = {
    "training": Training,
    "fine_tuning": FineTuning,
    "prompt_learning": PromptLearning,
    "conversion": Conversion,
    "evaluation": {
        EvalHarnessEvaluation: ["gpt3", "prompt_gpt3"],
        NeMoEvaluation: ["t5", "mt5", "prompt_t5", "prompt_mt5"]
    },
    "data_preparation": {
        PileDataPreparation: ["gpt3", "t5"],
        MC4DataPreparation: ["mt5"],
        CustomDataPreparation: ["generic"],
    }
}


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    requested_stages = cfg.get("stages")

    dependency = None
    for stage_name in requested_stages:
        stage_class = STR2STAGECLASS[stage_name]
        if isinstance(stage_class, dict):
            stage_config_choice = cfg.get(f"{stage_name}_config")
            choice_model_type = stage_config_choice.rsplit("/", 1)[0]
            for cls, model_types in stage_class.items():
                if choice_model_type in model_types:
                    stage_class = cls
                    break

        if dependency is not None:
            cfg[stage_name]["run"]["dependency"] = dependency
        stage = stage_class(cfg)
        job_id = stage.run()

        job_path = stage.get_job_path()
        command = " \\\n  ".join(sys.argv)
        with open(job_path.folder / "bignlp_cmd.log", "w") as f:
            f.write(command)

        if job_id:
            dependency = f"afterany:{job_id}"

if __name__ == "__main__":
    main()
