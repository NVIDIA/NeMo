import os

import hydra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    bignlp_path = cfg["bignlp_path"]
    training_config = cfg["training_config"]
    code_path = (
        "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    )
    training_config_path = os.path.join(bignlp_path, "conf/training")
    flags = f"--config-path={training_config_path} --config-name={training_config} "
    cmd = f'cd /opt/bignlp/NeMo; git rev-parse HEAD; cd /opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/opt/bignlp/NeMo/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7 python3 {code_path} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
