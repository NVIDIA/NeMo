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
from bignlp.utils.file_utils import download_single_file

class BigNLPStage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cluster = cfg.get("cluster_type")

        self.stage_name = None
        self.stage_cfg = None
        self.setup_stage_vars(cfg)
        self.job_name = self.stage_cfg.run.get("name")

    def setup_stage_vars(self, cfg):
        raise NotImplementedError

    def run(self) -> str:
        """Run current stage; returns job id"""
        # Setup folders and datasets
        self.setup_folder_and_data()
        # Save stage hydra config
        job_path = self.get_job_path()
        stage_cfg_path = BigNLPStage.save_stage_hydra_config(
            self.stage_cfg, job_path
        )
        # Make cluster parameters
        cluster_parameters = self._make_cluster_parameters(self.cluster)
        # Make command groups
        command_groups = self.make_stage_command_groups(stage_cfg_path)
        # Create launcher
        launcher = AutoLauncher(
            folder=job_path.folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def setup_folder_and_data(self) -> None:
        """Setup required folders and dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

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

        numa_override = [f"{k}={v}" for k, v in numa_cfg.items()]
        numa_command = [f"python3 -u {self._bignlp_path / 'bignlp/collections/numa_mapping.py'}", *numa_override]
        numa_command = " \\\n  ".join(numa_command)
        return [numa_command]

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
        data_dir = cfg.get("data_dir")
        base_results_dir = cfg.get("base_results_dir")
        mounts_string = f"{self._bignlp_path}:{self._bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"

        container_mounts = cfg.get("container_mounts")
        mounts_string += add_container_mounts(container_mounts)
        return mounts_string

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

        setup = None
        env_vars = self.get_env_vars()
        if env_vars:
            setup = [
                f"export {k}={v}" for k, v in env_vars.items()
            ]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "nodes": nodes,
            "time": time_limit,
            "ntasks_per_node": ntasks_per_node,
            "setup": setup,
        }
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = cluster_cfg.get("slurm")
            job_name_prefix = cluster_cfg.get("job_name_prefix")
            cluster_parameters = {
                **slurm_cfg
            }
            cluster_parameters.update({
                **shared_parameters,
                "container_image": container_image,
                "container_mounts": container_mounts,
            })
            cluster_parameters["job_name"] = job_name_prefix + cluster_parameters["job_name"]
        elif cluster == "bcp":
            cluster_parameters.update({
                **shared_parameters,
                "env_vars": env_vars,
            })
        elif cluster == "interactive":
            cluster_parameters.update(shared_parameters)

        return cluster_parameters

    def get_env_vars(self) -> Dict:
        env_vars = {
            k: v for k, v in self.cfg.get("env_vars")
            if v is not None
        }
        return env_vars

    def get_stage_config_choice(self):
        stage_config_choice = cfg.get(f"{self.stage_name}_config")
        choice_model_type = stage_config_choice.rsplit("/", 1)[0]
        choice_name = stage_config_choice.rsplit("/", 1)[1]
        return choice_model_type, choice_name

    @property
    def _bignlp_path(self) -> Path:
        return Path(self.cfg.get("bignlp_path"))

    @property
    def _nemo_code_path(self) -> Path:
        return Path("/opt/bignlp/NeMo")

    @property
    def _data_dir(self) -> Path:
        return Path(self.cfg.get("data_dir"))

    @property
    def _cuda_visible_devices(self) -> str:
        ntasks_per_node = self.stage_cfg.run.get("ntasks_per_node")
        if ntasks_per_node is None:
            ntasks_per_node = self.stage_cfg.trainer.get("devices", 0)
        return "CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7" \
            if ntasks_per_node == 8 else ""

    @property
    def _cuda_device_max_connections(self) -> str:
        model_cfg = self.stage_cfg.get("model")
        tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size", 1)
        return "CUDA_DEVICE_MAX_CONNECTIONS=1" \
            if tensor_model_parallel_size > 1 else ""

    @functools.lru_cache()
    def get_job_path(self, sub_stage=None) -> JobPaths:
        run_cfg = self.stage_cfg.get("run")
        results_dir = Path(run_cfg.get("results_dir")) # TODO: rename this to job dir in config
        if sub_stage is not None:
            results_dir = results_dir / sub_stage
        return JobPaths(results_dir, self.job_name)


class NeMoStage(BigNLPStage):

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
        choice_model_type, choice_name = self.get_stage_config_choice()
        code_path = self._get_nemo_code_path(choice_model_type)

        hydra_override = self._make_hydra_override()

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
        return hydra_override


class Training(NeMoStage):

    def setup_stage_vars(self, cfg):
        self.stage_name = "training"
        self.stage_cfg = cfg.get("training")

    def _make_hydra_override(self):
        hydra_override = []
        if self.cluster == "bcp":
            hydra_override += ["+cluster_type=BCP"]
        if self.stage_cfg.model.data.get("data_prefix", None) is None:
            preprocessed_dir = self.stage_cfg.run.get("preprocessed_dir")
            blending_alpha = self.stage_cfg.run.get("blending_alpha")
            auto_blend_command = \
                f"python3 {self._bignlp_path / 'bignlp/collections/auto_blend.py'} " \
                f"model_type={choice_model_type} " \
                f"preprocessed_dir={preprocessed_dir} " \
                f"blending_alpha={blending_alpha} "
            hydra_override += [f"model.data.data_prefix=\${{{auto_blend_command}}}"]
        return hydra_override

    def _get_nemo_code_path(self, model_type):
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "gpt3": self._nemo_code_path / "examples/nlp/language_modeling/megatron_gpt_pretraining.py"
        }
        return model_type_to_code_path[model_type]


class FineTuning(NeMoStage):

    def setup_stage_vars(self, cfg):
        self.stage_name = "fine_tuning"
        self.stage_cfg = cfg.get("fine_tuning")

    def setup_folder_and_data(self):
        super().setup_folder_and_data()

        # Prepare fine-tuning dataset
        # TODO: Update to squad
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")

        # GLUE for internal use
        from bignlp.utils.data_utils.download_glue import download_glue, TASKS_LOWER
        if task_name in TASKS_LOWER:
            download_glue(data_dir=os.path.join(data_dir, "glue_data"), tasks=task_name)

    def _get_nemo_code_path(self, model_type):
        if model_type == "gpt3":
            raise NotImplementedError("Fine-tuning is not supported in NeMo Megatron GPT-3 models.")
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
        }
        return model_type_to_code_path[model_type]


class PromptLearning(NeMoStage):

    def setup_stage_vars(self, cfg):
        self.stage_name = "prompt_learning"
        self.stage_cfg = cfg.get("prompt_learning")

    def setup_folder_and_data(self):
        # Setup folders
        super().setup_folder_and_data()

        # Prepare prompt learning dataset
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")
        # Prepare squad dataset
        if task_name == 'squad':
            data_dir = os.path.join(data_dir, "prompt_data")
            squad_dir = os.path.join(data_dir, "squad-v2.0")
            if not os.path.exists(squad_dir):
                os.makedirs(squad_dir)
                download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", squad_dir,
                                     "train-v2.0.json")
                download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", squad_dir,
                                     "dev-v2.0.json")
                preprocess_script = self._bignlp_path / "bignlp/utils/data_utils/prompt_learning_squad_preprocessing.py"
                os.system(
                    f"python {preprocess_script} "
                    f"--data-dir={squad_dir} "
                    f"--file-name='train-v2.0.json' "
                    f"--save-name-base='squad' "
                    f"--make-ground-truth "
                    f"--train-percent=0.8"
                )

    def _get_nemo_code_path(self, model_type):
        if model_type != "gpt3":
            raise NotImplementedError("Prompt Learning is only supported in NeMo Megatron GPT-3 models.")
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
        }
        return model_type_to_code_path[model_type]


class Conversion(BigNLPStage):

    def setup_stage_vars(self, cfg):
        self.stage_name = "conversion"
        self.stage_cfg = cfg.get("conversion")

    def _make_hparams_override_command(self):
        model_cfg = self.stage_cfg.get("model")
        hparams_file = model_cfg.get("hparams_file")
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
        override_command = [
            f"python3 -u {self._bignlp_path / 'bignlp/collections/hparams_override.py'}",
            *hparams_override
        ]
        override_command = " \\\n  ".join(override_command)
        return [override_command]

    def _make_checkpoint_search_command(self, **kwargs):
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return (
            f"python3 -u {self._bignlp_path / 'bignlp/collections/get_latest_checkpoint.py'} "
            f"{' '.join(checkpoint_override)}"
        )

    def make_stage_command_groups(self, stage_cfg_path) -> List[List[str]]:
        command_groups = [[], []]
        command_groups[0] += self._make_hparams_override_command()

        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")
        checkpoint_search_command = self._make_checkpoint_search_command(
            checkpoint_folder=model_cfg.get("checkpoint_folder"),
            checkpoint_name=model_cfg.get("checkpoint_name"),
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size"),
            pipeline_model_parallel_size=model_cfg.get("pipeline_model_parallel_size"),
        )
        command_groups[-1] += [f"export CKPT_NAME=$({checkpoint_search_command})"]

        nemo_file_name = run_cfg.get("nemo_file_name")
        hparams_override_file = self.get_job_path().results_folder / "hparams_override.yaml"
        nemo_file_path = self.get_job_path().results_folder / nemo_file_name
        code_path = self._nemo_code_path / "examples/nlp/language_modeling/megatron_ckpt_to_nemo.py"
        args = create_args_list(
            gpus_per_node=run_cfg.get("ntasks_per_node"),
            model_type=model_cfg.get("model_type"),
            checkpoint_folder=model_cfg.get("checkpoint_folder"),
            checkpoint_name="\${CKPT_NAME}",
            hparams_file=hparams_override_file,
            nemo_file_path=nemo_file_path,
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size"),
            pipeline_model_parallel_size=model_cfg.get("pipeline_model_parallel_size"),
        )
        args += ["--bcp"] if self.cluster == "bcp" else []

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class NeMoEvaluation(NeMoStage):
    
    def setup_stage_vars(self, cfg):
        self.stage_name = "evaluation"
        self.stage_cfg = cfg.get("evaluation")

    def _get_nemo_code_path(self, model_type):
        if model_type == "gpt3":
            raise ValueError("Evaluating GPT-3 models needs `EvalHarnessEvaluation` class.")
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_eval.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_seq2seq_eval.py",
        }
        return model_type_to_code_path[model_type]


class EvalHarnessEvaluation(BigNLPStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        choice_model_type, choice_name = self.get_stage_config_choice()
        self.prompt_evaluation = (choice_model_type == "prompt_gpt3")

    def setup_stage_vars(self, cfg):
        self.stage_name = "evaluation"
        self.stage_cfg = cfg.get("evaluation")

    def _make_download_command_string(self) -> str:
        data_dir = self.cfg.get("data_dir")
        cache_dir = os.path.join(data_dir, "eval_harness_data")
        run_cfg = self.stage_cfg.get("run")
        tasks = run_cfg.get("tasks")

        code_path = self._bignlp_path / "bignlp/collections/eval_harness/download.py"
        args = create_args_list(
            tasks=tasks,
            cache_dir=cache_dir,
        )
        download_command = [f"python3 -u {code_path}", *args]
        download_command_string = " \\\n  ".join(download_command)
        return download_command_string

    def make_stage_command_groups(self, stage_cfg_path) -> List[List[str]]:
        if self.prompt_evaluation:
            command_groups = [[]]
        else:
            command_groups = [[], []]
            command_groups[0] += [self._make_download_command_string()]

        data_dir = self.cfg.get("data_dir")
        cache_dir = os.path.join(data_dir, "eval_harness_data")
        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")

        code_path = self._bignlp_path / "bignlp/collections/eval_harness/evaluate.py"
        args = create_args_list(
            name=run_cfg.get("name"),
            model=model_cfg.get("model_type"),
            tasks=run_cfg.get("tasks"),
            cache_dir=cache_dir,
            output_path=self.get_job_path().results_folder,
            batch_size=model_cfg.get("eval_batch_size"),
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size"),
            pipeline_model_parallel_size=model_cfg.get("pipeline_model_parallel_size"),
            precision=model_cfg.get("precision"),
        )

        if self.prompt_evaluation:
            args += create_args_list(
                nemo_model=model_cfg.get("nemo_model"),
                prompt_dataset_paths=model_cfg.get("prompt_dataset_paths"),
            )
        else:
            args += create_args_list(
                vocab_file=model_cfg.get("vocab_file"),
                merge_file=model_cfg.get("merge_file"),
                checkpoint_folder=model_cfg.get("checkpoint_folder"),
                checkpoint_name=model_cfg.get("checkpoint_name"),
                hparams_file=model_cfg.get("hparams_file"),
            )

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class DataStage(BigNLPStage):
    """ Data Stage is for data preparations; it can have few sub_stages """

    def setup_stage_vars(self, cfg):
        self.stage_name = "data_preparation"
        self.stage_cfg = cfg.get("data_preparation")

    def _make_sub_stages(self):
        raise NotImplementedError

    def run(self) -> str:
        # Setup folders and datasets
        self.setup_folder_and_data()

        sub_stages = self._make_sub_stages()
        job_id = ""
        for sub_stage in sub_stages:
            # Save stage hydra config
            job_path = self.get_job_path(sub_stage)
            job_path.folder.mkdir(parents=True, exist_ok=True)

            stage_cfg_path = BigNLPStage.save_stage_hydra_config(
                self.stage_cfg, job_path
            )
            if job_id:
                dependency = f"aftercorr:{job_id}"
                self.stage_cfg["run"]["dependency"] = dependency

            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster, sub_stage)

            # Make command groups
            command_groups = self.make_stage_command_groups(stage_cfg_path, sub_stage)
            # Create launcher
            launcher = AutoLauncher(
                folder=job_path.folder,
                cluster=self.cluster,
                **cluster_parameters,
            )
            job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def make_stage_command_groups(
            self, stage_cfg_path, sub_stage=None,
    ) -> List[List[str]]:
        command_groups = [[]]

        command_groups[0] += self._make_sub_stage_command(sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        raise NotImplementedError

    def _make_cluster_parameters(
            self, cluster: str, sub_stage=None,
    ) -> Dict:
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")

        env_vars = self.get_env_vars()
        env_vars["NGC_ARRAY_TYPE"] = "MPIJob" # Required by BCP
        setup = [
            f"export {k}={v}" for k, v in env_vars.items()
        ]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "time": time_limit,
            "setup": setup,
        }
        private_parameters = self._make_private_cluster_parameters(
            cluster, sub_stage,
        )
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = cluster_cfg.get("slurm")
            job_name_prefix = cluster_cfg.get("job_name_prefix")
            cluster_parameters = {
                **slurm_cfg
            }
            cluster_parameters.update({
                **shared_parameters,
                **private_parameters,
            })
            cluster_parameters["job_name"] = job_name_prefix + cluster_parameters["job_name"]
        elif cluster == "bcp":
            cluster_parameters.update({
                **shared_parameters,
                **private_parameters,
            })
        elif cluster == "interactive":
            raise ValueError("Data preparation is not supported in interactive mode.")


class PileDataPreparation(BigNLPStage):

    def _make_sub_stages(self):
        sub_stages = []
        if self.stage_cfg.get("download_the_pile", False):
            sub_stages += ["download", "extract"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        download_vocab_url = data_cfg.get("download_vocab_url")
        download_merges_url = data_cfg.get("download_merges_url")
        vocab_save_dir = data_cfg.get("vocab_save_dir")
        merges_save_dir = data_cfg.get("merges_save_dir")
        # Download vocab
        if download_vocab_url is not None:
            assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
            utils.download_single_file(
                url=download_vocab_url,
                save_dir=vocab_save_dir,
                file_name="vocab.json" if download_vocab_url.endswith("json") else "vocab.txt",
            )
        # Download merges
        if download_merges_url is not None:
            assert merges_save_dir is not None, "merges_save_dir must be a valid path."
            utils.download_single_file(
                url=download_merges_url,
                save_dir=merges_save_dir,
                file_name="merges.txt",
            )

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        node_array_size = run_cfg.get("node_array_size")
        array = run_cfg.get("array")
        bcp_preproc_npernode = run_cfg.get("bcp_preproc_npernode")  \
            if sub_stage == "preprocess" else 1
        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": bcp_preproc_npernode,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

    def _make_sub_stage_command(self, sub_stage):
        pile_prep_path = self._bignlp_path / "bignlp/collections/dataprep_scripts/pile_dataprep"
        stage_to_code_path = {
            "download": pile_prep_path / "download.py",
            "extract": pile_prep_path / "extract.py",
            "preprocess": pile_prep_path / "preprocess.py",
        }
        choice_model_type, choice_name = self.get_stage_config_choice()

        code_path = stage_to_code_path[sub_stage]
        args = create_args_list(
            hydra=True,
            data_config=choice_name,
            cluster_type=self.cluster,
            bignlp_path=self._bignlp_path,
            data_dir=self._data_dir,
            the_pile_url=self.stage_cfg.get("the_pile_url"),
            file_numbers=self.stage_cfg.get("file_numbers"),
            rm_downloaded=self.stage_cfg.get("rm_downloaded"),
            rm_extracted=self.stage_cfg.get("rm_extracted"),
            tokenizer_type=self.stage_cfg.get("tokenizer_type"),
            vocab_save_dir=self.stage_cfg.get("vocab_save_dir"),
            merges_save_dir=self.stage_cfg.get("merges_save_dir"),
        )
        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class MC4DataPreparation(DataStage):

    def _make_sub_stages(self):
        sub_stages = []
        if self.stage_cfg.get("download_mc4", False):
            sub_stages += ["prepare", "download"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["setup_preprocess", "preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        download_vocab_url = data_cfg.get("download_vocab_url")
        download_tokenizer_url = data_cfg.get("download_tokenizer_url")
        vocab_save_dir = data_cfg.get("vocab_save_dir")
        tokenizer_save_dir = data_cfg.get("tokenizer_save_dir")

        if download_vocab_url is not None:
            assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_vocab_url,
                save_dir=vocab_save_dir,
                file_name="vocab.txt",
            )
        if download_tokenizer_url is not None:
            assert tokenizer_save_dir is not None, "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_tokenizer_url,
                save_dir=tokenizer_save_dir,
                file_name="mt5_tokenizer.model",
            )

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")

        node_array_size = run_cfg.get("node_array_size") \
            if sub_stage in ["download", "preprocess"] else 1
        array = f"0-{node_array_size-1}"
        if sub_stage == "preprocess":
            ntasks_per_node = run_cfg.get("workers_per_node")
            cpus_per_task = run_cfg.get("cpus_per_node") // ntasks_per_node
        else:
            ntasks_per_node = 1
            cpus_per_task = None

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
                "cpus_per_task": cpus_per_task,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": ntasks_per_node,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

    def _make_sub_stage_command(self, sub_stage):
        mc4_prep_path = self._bignlp_path / "bignlp/collections/dataprep_scripts/mc4_dataprep"
        stage_to_code_path = {
            "prepare": mc4_prep_path / "prepare.py",
            "download": mc4_prep_path / "download.py",
            "setup_preprocess": mc4_prep_path / "setup_preprocess.py",
            "preprocess": mc4_prep_path / "preprocess.py",
        }

        data_cfg = self.stage_cfg
        run_cfg = data_cfg.get("run")

        code_path = stage_to_code_path[sub_stage]
        if sub_stage == "prepare":
            args = create_args_list(
                data_path=data_cfg.get("mc4_dir"),
                git_lfs_path=data_cfg.get("git_lfs_path"),
                langauges=data_cfg.get("langauges"),
                node_array_size=run_cfg.get("node_array_size"),
                worker_mapping_file=data_cfg.get("download_worker_mapping"),
            )
            if data_cfg.get("use_cleaned_english"):
                args += ["--cleaned-en"]
        elif sub_stage == "download":
            args = create_args_list(
                c4_path=Path(data_cfg.get("mc4_dir")) / "c4",
                git_lfs_path=data_cfg.get("git_lfs_path"),
                worker_mapping_file=data_cfg.get("download_worker_mapping"),
            )
        elif sub_stage == "setup_preprocess":
            args = create_args_list(
                c4_path=Path(data_cfg.get("mc4_dir")) / "c4",
                soft_link_path=data_cfg.get("softlinks_dir"),
                langauges=data_cfg.get("langauges"),
                node_array_size=run_cfg.get("node_array_size"),
                workers_per_node=run_cfg.get("workers_per_node"),
                max_split_size=200,
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
            )
            if data_cfg.get("use_cleaned_english"):
                args += ["--cleaned-en"]
        else:
            assert sub_stage == "preprocess", f"Unknown substage {sub_stage}"
            args = create_args_list(
                output_path=data_cfg.get("preprocessed_dir"),
                workers_per_node=run_cfg.get("workers_per_node"),
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
                tokenizer_library=data_cfg.get("preprocessed_dir"),
                tokenizer_model=data_cfg.get("tokenizer_model"),
                dataset_impl="mmap",
                log_interval="2000",
                preproc_folder="store_true",
                apply_ftfy="store_true",
                workers=run_cfg.get("cpus_per_node") // run_cfg.get("workers_per_node"),
            )
            if data_cfg.get("rm_downloaded"):
                args += ["--rm-downloaded"]

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class CustomDataPreparation(DataStage):

    def _make_sub_stages(self):
        sub_stages = []
        if self.stage_cfg.get("train_tokenizer", False):
            sub_stages += ["train_tokenizer"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        # Setup preprocess data
        data_cfg = self.stage_cfg
        raw_dataset_files = data_cfg.get("raw_dataset_files")
        preprocess_worker_mapping = data_cfg.get("preprocess_worker_mapping")
        if data_cfg.get("preprocess_data", False):
            # Sort list of files in directory by size
            sorted_files = sorted(raw_dataset_files, key=lambda x: os.stat(x).st_size)
            file_sizes = [os.stat(x).st_size for x in sorted_files]

            avail_workers = nodes * workers_per_node
            distributed_files = [[] for _ in range(avail_workers)]
            distributed_size = [0] * avail_workers
            for f, file_size in zip(sorted_files, file_sizes):
                min_ind = distributed_size.index(min(distributed_size))
                distributed_files[min_ind].append(f)
                distributed_size[min_ind] += file_size

            output = [",".join(distributed_files[i]) for i in range(avail_workers)]
            output = "\n".join(output)
            with open(preprocess_worker_mapping, "w") as file:
                file.write(output)
            print(f" ****** Workers mapping saved to {preprocess_worker_mapping} ...")
            for i in range(avail_workers):
                print(
                    f"{i + 1:>4d} "
                    f"{distributed_size[i]:>7.2f}GB  "
                    f"{','.join([os.path.basename(file) for file in distributed_files[i]]):s}"
                )

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")

        if sub_stage == "preprocess":
            node_array_size = run_cfg.get("node_array_size")
            ntasks_per_node = run_cfg.get("workers_per_node")
            cpus_per_task = run_cfg.get("cpus_per_node") // ntasks_per_node
        else:
            node_array_size = 1
            ntasks_per_node = 1
            cpus_per_task = None
        array = f"0-{node_array_size - 1}"

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
                "cpus_per_task": cpus_per_task,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": ntasks_per_node,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

    def _make_sub_stage_command(self, sub_stage):
        data_cfg = self.stage_cfg
        if sub_stage == "train_tokenizer":
            bpe_save_dir = Path(data_cfg.get("bpe_save_dir"))
            bpe_save_dir.mkdir(parents=True, exist_ok=True)
            train_tokenizer_args = data_cfg.get("train_tokenizer_args")
            code_path = f"cd {bpe_save_dir} && spm_train"
            args = create_args_list(
                **train_tokenizer_args
            )
        else:
            assert sub_stage == "preprocess", f"Unknown substage {sub_stage}"
            code_path = self._bignlp_path / "bignlp/collections/dataprep_scripts/custom_dataprep/preprocess.py"
            args = create_args_list(
                output_path=data_cfg.get("preprocessed_dir"),
                workers_per_node=run_cfg.get("workers_per_node"),
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
                tokenizer_library=data_cfg.get("preprocessed_dir"),
                tokenizer_model=data_cfg.get("tokenizer_model"),
                dataset_impl="mmap",
                log_interval="2000",
                preproc_folder="store_true",
                apply_ftfy="store_true",
                workers=run_cfg.get("cpus_per_node") // run_cfg.get("workers_per_node"),
            )

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


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


def create_args_list(hydra=False, **kwargs):
    args = []
    for k, v in kwargs.items():
        if hydra:
            args.append(f"{k}={v}")
        else:
            # use "store_true" to add keys only args
            k = k.replace("_", "-")
            args.append(f"--{k}" if v == "store_true" else f"--{k}={v}")
    return args