import sys
import os
import subprocess

import hydra
import omegaconf
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

from bignlp.core.launchers import AutoLauncher
from bignlp.utils.job_utils import JobPaths

class BigNLPStage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cluster = cfg.get("cluster_type")

    def run(self) -> str:
        """Run current stage; returns job id"""
        # Setup folders and datasets
        self.setup()
        # Save stage hydra config
        job_path = self.get_job_path()
        stage_cfg_path = BigNLPStage.save_stage_hydra_config(
            self.stage_cfg, job_path
        )
        # Make command groups
        command_groups = self.make_stage_command_groups(stage_cfg_path)
        # Make cluster parameters
        cluster_parameters = self._make_cluster_parameters(self.cluster)
        # Create launcher
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def setup(self) -> None:
        """Setup required folders and dataset"""
        raise NotImplementedError

    @staticmethod
    def save_stage_hydra_config(stage_cfg, job_path) -> Path:
        """Save hydra config file for current stage"""
        _hydra_interpolation(stage_cfg)

        cfg_save_path = job_path.config_file
        omegaconf.OmegaConf.save(stage_cfg, cfg_save_path)
        return cfg_save_path

    def make_stage_command_groups(self, stage_cfg_path) -> List[List[str]]:
        raise NotImplementedError

    def _make_wandb_login_command(self) -> List[str]:
        """Login with w&b api key"""
        cfg = self.cfg
        wandb_cmd = ""
        if cfg.wandb_api_key_file is not None:
            with open(cfg.wandb_api_key_file, "r") as f:
                wandb_api_key = f.readline().rstrip()
            wandb_cmd = f"wandb login {wandb_api_key}"
        return [wandb_cmd]

    def _make_nemo_path_command(self) -> List[str]:
        """Extend nemo path to python path"""
        return [
            f"cd {self._nemo_code_path}",
            "git rev-parse HEAD",
            f'export PYTHONPATH={self._nemo_code_path}:\${{PYTHONPATH}}',
        ]

    def _make_numa_mapping_command(self) -> List[str]:
        cfg = self.cfg
        numa_cfg = cfg.get("numa_mapping")
        if not numa_cfg.get("enable"):
            return []

        bignlp_path = Path(cfg.get("bignlp_path"))
        numa_override = [f"{k}={v}" for k, v in numa_cfg.items()]
        return [
            f"python3 -u {bignlp_path / 'bignlp/collections/numa_mapping.py'} "
            f"{' '.join(numa_override)}"
        ]

    def _make_nsys_command_prefix(self, results_dir) -> str:
        model_cfg = self.stage_cfg.get("model")
        nsys_cfg = model_cfg.get("nsys_profile", None)
        nsys_prefix = ""
        if nsys_cfg is not None and nsys_cfg.get("enabled", False):
            profile_out_path = os.path.join(results_dir, "profile_logs")
            os.makedirs(profile_out_path, exist_ok=True)
            slurm_node = "\${SLURM_NODEID}"
            slurm_rank = "\${SLURM_PROCID}"
            slurm_jobid = "\${SLURM_JOB_ID}"
            nsys_prefix = f"nsys profile -s none " \
                          f"-t {','.join(nsys_cfg.trace)} " \
                          f"-o {profile_out_path}/profile_{slurm_jobid}_node{slurm_node}_rank{slurm_rank} " \
                          f"--force-overwrite true " \
                          f"--capture-range=cudaProfilerApi " \
                          f"--capture-range-end=stop"
        return nsys_prefix

    def _make_container_mounts_string(self) -> str:

        def add_container_mounts(container_mounts):
            mounts_str = ""
            if container_mounts is not None:
                assert isinstance(
                    container_mounts, omegaconf.listconfig.ListConfig
                ), "container_mounts must be a list."
                for mount in container_mounts:
                    if mount is not None and isinstance(mount, str):
                        mounts_str += f",{mount}" if ":" in mount else f",{mount}:{mount}"
            return mounts_str

        cfg = self.cfg
        bignlp_path = cfg.get("bignlp_path")
        data_dir = cfg.get("data_dir")
        base_results_dir = cfg.get("base_results_dir")
        mounts_string = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"

        container_mounts = cfg.get("container_mounts")
        mounts_string += add_container_mounts(container_mounts)
        return mounts_string

    @property
    def get_job_path(self):
        raise NotImplementedError

    @property
    def _nemo_code_path(self) -> Path:
        return Path("/opt/bignlp/NeMo")

    @property
    def _cuda_visible_devices(self) -> str:
        trainer_cfg = self.stage_cfg.get("trainer")
        ntasks_per_node = trainer_cfg.get("devices", 0) if trainer_cfg else 0
        return "CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7" \
            if ntasks_per_node == 8 else ""

    @property
    def _cuda_device_max_connections(self) -> str:
        model_cfg = self.stage_cfg.get("model")
        tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size", 1)
        return "CUDA_DEVICE_MAX_CONNECTIONS=1" \
            if tensor_model_parallel_size > 1 else ""


class Training(BigNLPStage):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.stage_name = None
        self.stage_cfg = None
        self.setup_stage_vars(cfg)
        self.job_name = self.stage_cfg.run.get("name")

    def setup_stage_vars(self, cfg):
        self.stage_name = "training"
        self.stage_cfg = cfg.get("training")

    def setup(self):
        # Setup folders
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def make_stage_command_groups(self, stage_cfg_path):
        # Training has one command group
        # Shared with fine-tuning and prompt learning
        command_groups = [[]]
        command_groups[0] += self._make_wandb_login_command()
        command_groups[0] += self._make_nemo_path_command()
        command_groups[0] += self._make_numa_mapping_command()

        core_command = [
            self._cuda_device_max_connections,
            self._cuda_visible_devices,
            self._make_nsys_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nemo_call_string(stage_cfg_path)
        ]
        core_command_string = " ".join(core_command)
        command_groups[0] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups

    def _make_nemo_call_string(self, stage_cfg_path) -> str:
        cfg = self.cfg
        stage_config_choice = cfg.get(f"{self.stage_name}_config")
        choice_model_type = stage_config_choice.rsplit("/", 1)[0]
        choice_name = stage_config_choice.rsplit("/", 1)[1]

        code_path = self._get_nemo_code_path(choice_model_type)

        hydra_override = self._hydra_override()

        command = [
            f"python3 -u {code_path} ",
            f"--config-path={stage_cfg_path.parents[0]}",
            f"--config-name={stage_cfg_path.name}",
            *hydra_override
        ]
        command_string = " \\\n  ".join(command)
        return command_string

    def _make_hydra_override(self):
        hydra_override = []
        if self.cluster == "bcp":
            hydra_override += ["+cluster_type=BCP"]
        if self.stage_name.model.data.get("data_prefix", None) is None:
            bignlp_path = Path(cfg.get("bignlp_path"))
            preprocessed_dir = self.stage_cfg.run.get("preprocessed_dir")
            blending_alpha = self.stage_cfg.run.get("blending_alpha")
            auto_blend_command = \
                f"python3 {bignlp_path / 'bignlp/collections/auto_blend.py'} " \
                f"model_type={choice_model_type} " \
                f"preprocessed_dir={preprocessed_dir} " \
                f"blending_alpha={blending_alpha} "
            hydra_override += [f"model.data.data_prefix=\${{{auto_blend_command}}}"]
        return hydra_override

    def _make_cluster_parameters(self, cluster: str) -> Dict:
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        nodes = run_cfg.get("nodes")
        if nodes is None:
            nodes = stage_cfg.get("trainer").get("num_nodes")
        ntasks_per_node = run_cfg.get("ntasks_per_node")
        if ntasks_per_node is None:
            ntasks_per_node = stage_cfg.get("trainer").get("devices")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        cluster_parameters = {}
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = cluster_cfg.get("slurm")
            job_name_prefix = cluster_cfg.get("job_name_prefix")
            slurm_job_name = job_name_prefix + job_name
            cluster_parameters = {
                **slurm_cfg
            }
            cluster_parameters.update({
                "job_name": slurm_job_name,
                "time": time_limit,
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
                "container_image": container_image,
                "container_mounts": container_mounts,
            })
        elif cluster == "bcp":
            cluster_parameters.update({
                "job_name": job_name,
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
            })
        elif cluster == "interactive":
            cluster_parameters.update({
                "job_name": job_name,
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
            })

        return cluster_parameters

    def _get_nemo_code_path(self, model_type):
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "gpt3": self._nemo_code_path / "examples/nlp/language_modeling/megatron_gpt_pretraining.py"
        }
        return model_type_to_code_path[model_type]

    @functools.lru_cache()
    def get_job_path(self) -> JobPaths:
        run_cfg = self.stage_cfg.get("run")
        results_dir = run_cfg.get("results_dir") # TODO: rename this to job dir in config
        return JobPaths(results_dir, self.job_name)


class FineTuning(Training):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        self.stage_name = "fine_tuning"
        self.stage_cfg = cfg.get("fine_tuning")

    def setup(self):
        # Setup folders
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

        # Prepare fine-tuning dataset
        # TODO: Update to squad
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")

        # GLUE for internal use
        from bignlp.utils.data_utils.download_glue import download_glue, TASKS_LOWER
        if task_name in TASKS_LOWER:
            download_glue(data_dir=os.path.join(data_dir, "glue_data"), tasks=task_name)

    def _make_hydra_override(self):
        hydra_override = []
        if self.cluster == "bcp":
            hydra_override += ["+cluster_type=BCP"]
        return hydra_override

    def _get_nemo_code_path(self, model_type):
        if model_type == "gpt3":
            raise NotImplementedError("Fine-tuning is not supported in NeMo Megatron GPT-3 models.")
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
        }
        return model_type_to_code_path[model_type]


class PromptLearning(Training):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        self.stage_name = "prompt_learning"
        self.stage_cfg = cfg.get("prompt_learning")

    def setup(self):
        # Setup folders
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

        # Prepare prompt learning dataset
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")
        # Prepare squad dataset
        if task_name == 'squad':
            data_dir = os.path.join(data_dir, "prompt_data")
            squad_dir = os.path.join(data_dir, "squad-v2.0")
            if not os.path.exists(squad_dir):
                os.makedirs(squad_dir)
                bignlp_path = Path(self.cfg.get("bignlp_path"))
                download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", squad_dir,
                                     "train-v2.0.json")
                download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", squad_dir,
                                     "dev-v2.0.json")
                preprocess_script = bignlp_path / "bignlp/utils/data_utils/prompt_learning_squad_preprocessing.py"
                os.system(
                    f"python {preprocess_script} "
                    f"--data-dir={squad_dir} "
                    f"--file-name='train-v2.0.json' "
                    f"--save-name-base='squad' "
                    f"--make-ground-truth "
                    f"--train-percent=0.8"
                )

    def _make_hydra_override(self):
        hydra_override = []
        if self.cluster == "bcp":
            hydra_override += ["+cluster_type=BCP"]
        return hydra_override

    def _get_nemo_code_path(self, model_type):
        if model_type != "gpt3":
            raise NotImplementedError("Prompt Learning is only supported in NeMo Megatron GPT-3 models.")
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
        }
        return model_type_to_code_path[model_type]


class Conversion(BigNLPStage):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.stage_name = None
        self.stage_cfg = None
        self.setup_stage_vars(cfg)
        self.job_name = self.stage_cfg.run.get("name")

    def setup_stage_vars(self, cfg):
        self.stage_name = "conversion"
        self.stage_cfg = cfg.get("conversion")

    def setup(self) -> None:
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def _make_hparams_override_command(self):
        cfg = self.cfg
        model_cfg = self.stage_cfg.get("model")
        hparams_file = model_cfg.get("hparams_file")
        bignlp_path = Path(cfg.get("bignlp_path"))
        vocab_file = model_cfg.get("vocab_file")
        merge_file = model_cfg.get("merge_file")
        tokenizer_model = model_cfg.get("tokenizer_model")
        override_configs = {
            "hparams_file": hparams_file,
            "output_path": self.get_job_path().results_folder,
            "vocab_file": vocab_file,
            "merge_file": merge_file,
            "tokenizer_model": tokenizer_model,
        }
        hparams_override = [f"{k}={v}" for k, v in override_configs.items()]
        return [
            f"python3 -u {bignlp_path / 'bignlp/collections/hparams_override.py'} "
            f"{' '.join(hparams_override)}"
        ]

    def _make_checkpoint_search_command(self, **kwargs):
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return [
            f"python3 -u {bignlp_path / 'bignlp/collections/get_latest_checkpoint.py'} "
            f"{' '.join(checkpoint_override)}"
        ]

    def make_stage_command_groups(self, stage_cfg_path) -> List[List[str]]:
        command_groups = [[], []]
        command_groups[1] = [self._make_hparams_override_command()]

        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")
        nemo_file_name = run_cfg.get("nemo_file_name")
        gpus_per_node = run_cfg.get("ntasks_per_node")
        model_type = model_cfg.get("model_type")
        checkpoint_folder = model_cfg.get("checkpoint_folder")
        checkpoint_name = model_cfg.get("checkpoint_name")
        tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size")
        pipeline_model_parallel_size = model_cfg.get("pipeline_model_parallel_size")
        hparams_override_file = self.get_job_path().results_folder / "hparams_override.yaml"
        nemo_file_path = self.get_job_path().results_folder / nemo_file_name

        checkpoint_search_command = self._make_checkpoint_search_command(
            checkpoint_folder=checkpoint_folder,
            checkpoint_name=checkpoint_name,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )
        command_groups[-1] += f"export CKPT_NAME=$({checkpoint_search_command})"

        code_path = "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py"
        args = [
            f"--gpus_per_node={gpus_per_node}",
            f"--model_type={model_type}",
            f"--checkpoint_folder={checkpoint_folder}",
            f"--checkpoint_name=\${{CKPT_NAME}}",
            f"--hparams_file={hparams_override_file}",
            f"--nemo_file_path={nemo_file_path}",
            f"--tensor_model_parallel_size={tensor_model_parallel_size}",
            f"--pipeline_model_parallel_size={pipeline_model_parallel_size}",
        ]
        args += ["--bcp"] if self.cluster == "bcp" else []

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups



def clean_command_groups(command_groups):
    for ind, command_group in enumerate(command_groups):
        command_groups[ind] = [c for c in command_group if c]
    return command_groups


def _hydra_interpolation(cfg) -> None:
    """Interpolate hydra config values, bypassing lazy interpolation"""
    def interpolate(cfg):
        if isinstance(cfg, omegaconf.dictconfig.DictConfig):
            for k, v in cfg.items():
                cfg[k] = interpolate(v)
        elif isinstance(cfg, omegaconf.listconfig.ListConfig):
            for i, v in enumerate(cfg):
                cfg[i] = interpolate(v)
        return cfg
    interpolate(cfg)