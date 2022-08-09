import sys
import os
import subprocess

import hydra
import omegaconf
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

from bignlp.core.launchers import AutoLauncher
from bignlp.core.stages import BigNLPStage
from bignlp.core.stages import clean_command_groups, create_args_list
from bignlp.utils.job_utils import JobPaths
from bignlp.utils.file_utils import download_single_file


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

        return cluster_parameters


class PileDataPreparation(DataStage):

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
            download_single_file(
                url=download_vocab_url,
                save_dir=vocab_save_dir,
                file_name="vocab.json" if download_vocab_url.endswith("json") else "vocab.txt",
            )
        # Download merges
        if download_merges_url is not None:
            assert merges_save_dir is not None, "merges_save_dir must be a valid path."
            download_single_file(
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