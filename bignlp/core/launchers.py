import functools
import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import warnings
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

import bignlp.utils.job_utils as job_utils
from bignlp.core.logger import logger

BIGNLP_CI = os.getenv("BIGNLP_CI", "False").lower() in ("true", "t", "1")
BIGNLP_DEBUG = os.getenv("BIGNLP_DEBUG", "False").lower() in ("true", "t", "1")
BIGNLP_MEMORY_MEASURE = os.getenv("BIGNLP_MEMORY_MEASURE", "False").lower() in ("true", "t", "1")

class AutoLauncher:
    def __init__(
            self, folder: Union[str, Path], job_name: str, cluster: Optional[str] = None, **kwargs: Any
    ) -> None:
        self.cluster = cluster or self.which()
        self.cluster = self.cluster.lower()

        launchers = self.get_launchers()
        if self.cluster not in launchers:
            raise ValueError(f"AutoLauncher doesn't know any cluster named {self.cluster}")

        self._launcher = launchers[self.cluster](folder, job_name, **kwargs)

    def launch(
            self, command_groups: List[List[str]]
    ) -> str:
        job_id = self._launcher.launch(command_groups)
        return job_id

    @staticmethod
    def which() -> str:
        """Returns what is the detected cluster."""
        raise NotImplementedError

    @staticmethod
    def get_launchers():
        return {
            "bcm": SlurmLauncher,
            "bcp": BCPLauncher,
            "interactive": InteractiveLauncher,
        }


class Launcher:
    def __init__(self, folder: Union[Path, str], job_name: str):
        self.folder = folder
        self.job_name = job_name

    def launch(
            self, command_groups: List[List[str]]
    ) -> str:
        submission_file_path = self._make_submission_file(
            command_groups
        )
        logger.info(f"Job {self.job_name} submission file created at '{submission_file_path}'")

        job_id = ""
        if not BIGNLP_DEBUG:
            job_id = self._submit_command(submission_file_path)
            if job_id:
                logger.info(f"Job {self.job_name} submitted with Job ID {job_id}")
                with open(self.folder / "launcher.log", "w") as f:
                    f.write(f"Submitted batch job {job_id}")

        return job_id

    def _submit_command(
        self, submission_file_path: Path
    ) -> str:
        """Submits a set of command groups to the cluster"""
        raise NotImplementedError

    def _make_submission_file(
            self, command_groups: List[List[str]]
    ) -> Path:
        job_paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        folder = job_paths.folder
        folder.mkdir(parents=True, exist_ok=True)

        submission_file_path = job_paths.submission_file
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command_groups))
        return submission_file_path


class InteractiveLauncher(Launcher):
    def __init__(
            self, folder: Union[Path, str], job_name: str, **kwargs: Any
    ) -> None:
        super().__init__(folder, job_name)
        self.parameters = kwargs

    def _submit_command(
        self, submission_file_path: Path
    ) -> str:
        """Submits a set of command groups to the cluster"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        job_utils.CommandFunction(command_list, ret_stdout=False, verbose=False)()  # explicit errors
        return ""

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        return ["bash", str(submission_file_path)]

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        nodes = self.parameters.get("nodes", 1)
        ntasks_per_node = self.parameters.get("ntasks_per_node", 1)
        assert nodes == 1, "Multi-node is not supported in interactive mode."

        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        time_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
        stdout = str(paths.stdout).replace("_%j", f"_{time_tag}")

        # now create
        lines = ["#!/bin/bash", ""]

        # environment setup:
        setup = self.parameters.get("setup", None)
        if setup is not None:
            lines += ["", "# setup"] + setup

        for group_ind, command_group in enumerate(command_groups):
            command = ";\n  ".join(command_group)
            command = command.replace(
                "python3 -u", f"torchrun --nproc_per_node={ntasks_per_node}"
            )

            lines += [
                "",
                f"# command {group_ind + 1}",
                f"bash -c \"",
                f"  {command} \" 2>&1 | tee -a {stdout}",
                "",
            ]
        return "\n".join(lines)


class BCPLauncher(Launcher):
    """BCP Job launcher"""
    def __init__(
            self, folder: Union[Path, str], job_name: str, **kwargs: Any
    ) -> None:
        super().__init__(folder, job_name)
        self.parameters = kwargs
        self.parameters = self._convert_parameters(self.parameters)

    @classmethod
    def _equivalence_dict(cls):
        return {
            "name": "job_name",
            "nodes": "nnodes",
            "tasks_per_node": "npernode",
            "ntasks_per_node": "npernode",
            "bcp_launcher": "launcher",
        }

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # replace type in some cases
        eq_dict = self._equivalence_dict()
        if eq_dict is not None:
            params = {eq_dict.get(k, k): v for k, v in params.items()}
        return params

    def _submit_command(
        self, submission_file_path: Path
    ) -> str:
        """Submits a set of command groups to the cluster"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        job_utils.CommandFunction(command_list, ret_stdout=False, verbose=False)()  # explicit errors
        return ""

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        return ["bash", str(submission_file_path)]

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        time_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
        stdout = str(paths.stdout).replace("_%j", f"_{time_tag}")

        nnodes = self.parameters.get("nnodes", 1)
        npernode = self.parameters.get("npernode", 1)
        launcher = self.parameters.get("launcher")
        launcher_flags = ""
        if launcher is not None:
            launcher_flags = f"--launcher {launcher}"
        env_vars = self.parameters.get("env_vars")
        env_flags = ""
        if env_vars is not None:
            env_flags = [f"--env '{k}={v}'" for k, v in env_vars.items()]
            env_flags = " ".join(env_flags)

        # now create
        lines = ["#!/bin/bash", ""]

        # environment setup:
        setup = self.parameters.get("setup", None)
        if setup is not None:
            lines += ["", "# setup"] + setup

        # Add pause_and_prime_dns_connection to command groups on BCP
        bignlp_path = Path("/opt/bignlp/bignlp-scripts") # Hard code path on BCP
        pause_and_prime_dns_connection_command = (
            f"python3 -u {bignlp_path / 'bignlp/collections/pause_and_prime_dns_connections.py'}"
        )
        command_groups = [[pause_and_prime_dns_connection_command]] + command_groups

        for group_ind, command_group in enumerate(command_groups):
            command = ";\n  ".join(command_group)
            lines += [
                "",
                f"# command {group_ind + 1}",
                f"bcprun --nnodes {nnodes} --npernode {npernode} "
                f"{launcher_flags} {env_flags} --cmd \"",
                f"  {command} \" 2>&1 | tee -a {stdout}",
                "",
            ]
        return "\n".join(lines)


class SlurmLauncher(Launcher):
    """Slurm job launcher
    This class is used to hold the parameters to run a job on slurm.
    In practice, it will create a batch file in the specified directory for each job,
    ...

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.

    **kwargs: Any
            See slurm documentation for most parameters.
            Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
            Below are the parameters that differ from slurm documentation:

            setup: list
                a list of command to run in sbatch before running srun
    """

    def __init__(
            self, folder: Union[Path, str], job_name: str, **kwargs: Any
    ) -> None:
        super().__init__(folder, job_name)
        self.parameters = {}
        self._update_parameters(job_name=job_name, **kwargs)

        if shutil.which("srun") is None and not BIGNLP_DEBUG:
            raise RuntimeError('Could not detect "srun", are you indeed on a slurm cluster?')

    @classmethod
    def _equivalence_dict(cls):
        return {
            "name": "job_name",
            "timeout_min": "time",
            "mem_gb": "mem",
            "nodes": "nodes",
            "cpus_per_task": "cpus_per_task",
            "gpus_per_node": "gpus_per_node",
            "tasks_per_node": "ntasks_per_node",
        }

    @classmethod
    def _valid_parameters(cls) -> Set[str]:
        """Parameters that can be set through update_parameters"""
        return set(_get_default_parameters())

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # replace type in some cases
        eq_dict = self._equivalence_dict()
        if eq_dict is not None:
            params = {eq_dict.get(k, k): v for k, v in params.items()}
        if "mem" in params:
            params["mem"] = _convert_mem(params["mem"])
        return params

    def _update_parameters(self, **kwargs: Any) -> None:
        """Updates sbatch submission file parameters

        Parameters
        ----------
        See slurm documentation for most parameters.
        Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
        Below are the parameters that differ from slurm documentation:

        setup: list
            a list of command to run in sbatch before running srun

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values

        """
        defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {in_valid_parameters}\nValid parameters are:\n  - {string}"
            )

        self.parameters.update(kwargs)
        self.parameters = self._convert_parameters(self.parameters)

    def _submit_command(
        self, submission_file_path: Path
    ) -> str:
        """Submits a set of command groups to the cluster"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        output = job_utils.CommandFunction(command_list, verbose=False)()  # explicit errors

        job_id = ""
        if output:
            job_id = self._get_job_id_from_submission_command(output)
        return job_id

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        return _make_sbatch_string(
            command_groups=command_groups,
            folder=self.folder,
            **self.parameters
        )

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        return ["sbatch", str(submission_file_path)]

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        """Returns the job ID from the output of sbatch string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"job (?P<id>[0-9]+)", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of sbatch output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")


@functools.lru_cache()
def _get_default_parameters() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_sbatch_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command_groups", "folder"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_sbatch_string(
    command_groups: List[List[str]],
    folder: Union[str, Path],
    job_name: str = "bignlp",
    partition: Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: Optional[int] = None,
    cpus_per_task: Optional[int] = None,
    cpus_per_gpu: Optional[int] = None,
    num_gpus: Optional[int] = None,  # legacy
    gpus_per_node: Optional[int] = None,
    gpus_per_task: Optional[int] = None,
    qos: Optional[str] = None,  # quality of service
    setup: Optional[List[str]] = None,
    mem: Optional[str] = None,
    mem_per_gpu: Optional[str] = None,
    mem_per_cpu: Optional[str] = None,
    dependency: Optional[str] = None,
    comment: Optional[str] = None,
    constraint: Optional[str] = None,
    exclude: Optional[str] = None,
    account: Optional[str] = None,
    gres: Optional[str] = None,
    exclusive: Optional[Union[bool, str]] = None,
    array: Optional[str] = None,
    stderr_to_stdout: bool = False,
    container_image: Optional[str] = None,
    container_mounts: Optional[str] = None,
    additional_parameters: Optional[Dict[str, Any]] = None,
    srun_args: Optional[Iterable[str]] = None,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    command_groups:
        each command group will be assigned one srun
    folder: str/Path
        folder where print logs and error logs will be written
    setup: list
        a list of command to run in sbatch before running srun
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in bignlp.
        Eg: {"mail-user": "blublu@nvidia.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command_groups",
        "additional_parameters",
        "setup",
        "stderr_to_stdout",
        "container_image",
        "container_mounts",
        "srun_args",
    ]
    parameters = {k: v for k, v in locals().items() if v is not None and k not in nonslurm}
    # rename and reformat parameters

    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn('"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")')
    # add necessary parameters
    job_name = parameters.get("job_name")
    paths = job_utils.JobPaths(folder=folder, job_name=job_name)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)

    if array is not None:
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")

    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")

    if BIGNLP_CI: # Override output file for slurm
        parameters["output"] = parameters["error"] = str(paths.folder / "slurm_%j.out")
        stdout = stderr = parameters["output"]

    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_sbatch_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup

    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
    stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
    container_flags = ["--container-image", container_image] if container_image else []
    container_flags += ["--container-mounts", container_mounts] if container_mounts else []
    if srun_args is None:
        srun_args = []

    if BIGNLP_MEMORY_MEASURE:
        srun_args += ["--overlap"]

        mem_stdout = stdout.replace("_%j", "_mem_%j")
        mem_stdout = mem_stdout.replace("_%A_%a", "_mem_%A_%a")
        mem_csv_out = folder / "gpu_memory_measure.csv"
        mem_srun_cmd = shlex.join([
            "srun", "--ntasks=1", "--ntasks-per-node=1", "--output", mem_stdout, *container_flags, *srun_args
        ])
        lines += [
            "",
            "# run memory measure",
            f"{mem_srun_cmd} \\",
            f"  nvidia-smi --query-gpu=timestamp,index,,memory.total,memory.free,memory.used \\",
            f"  --format=csv -l 1 > {mem_csv_out} & ",
            "",
        ]

    for group_ind, command_group in enumerate(command_groups):
        srun_cmd = shlex.join(["srun", "--output", stdout, *stderr_flags, *container_flags, *srun_args])
        command = ";\n  ".join(command_group)
        lines += [
            "",
            f"# command {group_ind + 1}",
            f"{srun_cmd} bash -c \"",
            f"  {command} \"",
            "",
        ]
    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    """Convert non-integer mem_gb to unit MB"""
    if mem_gb == int(mem_gb):
        if int(mem_gb) == 0:
            return "0"
        return f"{int(mem_gb)}GB"
    return f"{int(mem_gb * 1024)}MB"


def _as_sbatch_flag(key: str, value: Any) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"
