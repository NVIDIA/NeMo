"""This module contains the code to statup triton inference servers."""
import logging
import os
import shutil
import subprocess
import typing

from jinja2 import Environment, FileSystemLoader

_ENSEMBLE_MODEL_CONFIGS = "/opt/NeMo/nemo/deploy/tensorrt_llm_backend/configs/"
_DEFAULT_MODEL_DIR = "/opt/ensemble"
_TRITON_BIN = "/opt/tritonserver/bin/tritonserver"
_MPIRUN_BIN = "/usr/local/mpi/bin/mpirun"
_LOGGER = logging.getLogger(__name__)


class ModelServer:
    """Abstraction of a multi-gpu triton inference server cluster."""

    def __init__(self, model=None, http: bool = False, max_batch_size: int = 128, model_repo_dir: str = None) -> None:
        """Initialize the model server."""
        self._model = model
        self._http = http
        self._max_batch_size = max_batch_size
        self._model_repo_dir = model_repo_dir
        self._render_model_templates()

    @property
    def _decoupled_mode(self) -> str:
        """Indicate if the Triton models should be hosted in decoupled mode for streaming."""
        return "true" if not self._http else "false"

    @property
    def _allow_http(self) -> str:
        """Indicate if Triton should allow http connections."""
        return "true" if self._http else "false"

    @property
    def _allow_grpc(self) -> str:
        """Inidicate if Triton should allow grpc connections."""
        return "true" if not self._http else "false"

    @property
    def _gpt_model_type(self) -> str:
        return "inflight_fused_batching"

    @property
    def model_repository(self) -> str:
        """Return the triton model repository."""
        return _DEFAULT_MODEL_DIR if self._model_repo_dir is None else self._model_repo_dir

    def _triton_server_cmd(self, rank: int) -> typing.List[str]:
        """Generate the command to start a single triton server of given rank."""
        return [
            "-n",
            "1",
            _TRITON_BIN,
            "--allow-http",
            self._allow_http,
            "--allow-grpc",
            self._allow_grpc,
            "--model-repository",
            self.model_repository,
            "--disable-auto-complete-config",
            f"--backend-config=python,shm-region-prefix-name=prefix{rank}_",
            ":",
        ]

    @property
    def _cmd(self) -> typing.List[str]:
        """Generate the full command."""
        cmd = [_MPIRUN_BIN]
        # TODO: multigpu
        for rank in range(1):
            cmd += self._triton_server_cmd(rank)
        return cmd

    @property
    def _env(self) -> typing.Dict[str, str]:
        """Return the environment variable for the triton inference server."""
        env = dict(os.environ)
        env["TRT_ENGINE_DIR"] = self._model.model_dir
        env["TOKENIZER_DIR"] = self._model.model_dir
        if os.getuid() == 0:
            _LOGGER.warning(
                "Triton server will be running as root. It is recommended that you don't run this container as root."
            )
            env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
            env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        return env

    def _render_tensorrt_llm_template(self, env):
        template_path = os.path.join("tensorrt_llm", "config.pbtxt.j2")
        output_path = os.path.join(self.model_repository, "tensorrt_llm", "config.pbtxt")

        template = env.get_template(template_path)

        with open(output_path, "w", encoding="UTF-8") as out:
            template_args = {
                "engine_dir": self._model.model_dir,
                "decoupled_mode": self._decoupled_mode,
                "gpt_model_type": self._gpt_model_type,
                "max_batch_size": self._max_batch_size,
            }
            out.write(template.render(**template_args))

    def _render_ensemble_template(self, env):
        template_path = os.path.join("ensemble", "config.pbtxt.j2")
        output_path = os.path.join(self.model_repository, "ensemble", "config.pbtxt")

        template = env.get_template(template_path)

        with open(output_path, "w", encoding="UTF-8") as out:
            template_args = {
                "max_batch_size": self._max_batch_size,
            }
            out.write(template.render(**template_args))

    def _render_postprocessing_template(self, env):
        template_path = os.path.join("postprocessing", "config.pbtxt.j2")
        output_path = os.path.join(self.model_repository, "postprocessing", "config.pbtxt")

        template = env.get_template(template_path)

        with open(output_path, "w", encoding="UTF-8") as out:
            template_args = {
                "max_batch_size": self._max_batch_size,
            }
            out.write(template.render(**template_args))

    def _render_preprocessing_template(self, env):
        template_path = os.path.join("preprocessing", "config.pbtxt.j2")
        output_path = os.path.join(self.model_repository, "preprocessing", "config.pbtxt")

        template = env.get_template(template_path)

        with open(output_path, "w", encoding="UTF-8") as out:
            template_args = {
                "max_batch_size": self._max_batch_size,
            }
            out.write(template.render(**template_args))

    def _render_model_templates(self) -> None:
        """Render and Jinja templates in the model directory."""
        try:
            # Copy the entire folder and its contents recursively
            shutil.copytree(_ENSEMBLE_MODEL_CONFIGS, self.model_repository)
            print(f"Folder '{_ENSEMBLE_MODEL_CONFIGS}' copied to '{self.model_repository}' successfully.")
        except shutil.Error as e:
            print(f"Error copying folder: {e}")
        except OSError as e:
            print(f"Error: {e}")

        env = Environment(
            loader=FileSystemLoader(searchpath=self.model_repository), autoescape=False,
        )  # nosec; all the provided values are from code, not the user

        self._render_ensemble_template(env)
        self._render_preprocessing_template(env)
        self._render_postprocessing_template(env)
        self._render_tensorrt_llm_template(env)

    def run(self) -> None:
        """Start the triton inference server."""
        cmd = self._cmd
        env = self._env

        _LOGGER.debug("Starting triton with the command: %s", " ".join(cmd))
        _LOGGER.debug("Starting triton with the env vars: %s", repr(env))
        try:
            self.proc = subprocess.Popen(cmd, env=env)
        except subporcess.CalledProcessError as e:
            _LOGGER.error("Error:", e)

    def stop(self) -> None:
        if self.proc != None:
            self.proc.terminate()
