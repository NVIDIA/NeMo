"""This module contains the code to statup triton inference servers."""
import logging
import os
import subprocess
import typing

from jinja2 import Environment, FileSystemLoader

_ENSEMBLE_MODEL_DIR = "/opt/NeMo/nemo/deploy/tensorrt_llm_backend/configs/ensemble_models/llama"
_TRITON_BIN = "/opt/tritonserver/bin/tritonserver"
_MPIRUN_BIN = "/usr/local/mpi/bin/mpirun"
_LOGGER = logging.getLogger(__name__)


class ModelServer:
    """Abstraction of a multi-gpu triton inference server cluster."""

    def __init__(self, model=None, http: bool = False) -> None:
        """Initialize the model server."""
        self._model = model
        self._http = http
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
        return _ENSEMBLE_MODEL_DIR

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

    def _render_model_templates(self) -> None:
        """Render and Jinja templates in the model directory."""
        env = Environment(
            loader=FileSystemLoader(searchpath=self.model_repository),
            autoescape=False,
        )  # nosec; all the provided values are from code, not the user

        template_path = os.path.join("tensorrt_llm", "config.pbtxt.j2")
        output_path = os.path.join(
            self.model_repository, "tensorrt_llm", "config.pbtxt"
        )

        template = env.get_template(template_path)

        with open(output_path, "w", encoding="UTF-8") as out:
            template_args = {
                "engine_dir": self._model.model_dir,
                "decoupled_mode": self._decoupled_mode,
                "gpt_model_type": self._gpt_model_type,
            }
            out.write(template.render(**template_args))

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
        print("##############$$$$$$$$$$$$")


    def stop(self) -> None:
        if self.proc != None:
            self.proc.terminate()

