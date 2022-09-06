import io
import sys
import select
import subprocess

from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union


class JobPaths:
    """Creates paths related to the slurm job and its submission"""

    def __init__(
        self, folder: Union[Path, str], job_name: str,
    ) -> None:
        self._folder = Path(folder).expanduser().absolute()
        self._job_name = job_name

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def results_folder(self) -> Path:
        return self._folder / 'results'

    @property
    def submission_file(self) -> Path:
        return Path(self.folder / f"{self._job_name}_submission.sh")

    @property
    def config_file(self) -> Path:
        return Path(self.folder / f"{self._job_name}_hydra.yaml")

    @property
    def stderr(self) -> Path:
        return Path(self.folder / f"log-{self._job_name}_%J.err")

    @property
    def stdout(self) -> Path:
        return Path(self.folder / f"log-{self._job_name}_%J.out")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.folder})"


class CommandFunction:
    """Wraps a command as a function in order to make sure it goes through the
    pipeline and notify when it is finished.
    The output is a string containing everything that has been sent to stdout.
    WARNING: use CommandFunction only if you know the output won't be too big !
    Otherwise use subprocess.run() that also streams the outputto stdout/stderr.

    Parameters
    ----------
    command: list
        command to run, as a list
    verbose: bool
        prints the command and stdout at runtime
    cwd: Path/str
        path to the location where the command must run from

    Returns
    -------
    str
       Everything that has been sent to stdout
    """

    def __init__(
        self,
        command: List[str],
        verbose: bool = True,
        ret_stdout: bool = True,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(command, list):
            raise TypeError("The command must be provided as a list")
        self.command = command
        self.verbose = verbose
        self.ret_stdout = ret_stdout
        self.cwd = None if cwd is None else str(cwd)
        self.env = env

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Call the cammand line with addidional arguments
        The keyword arguments will be sent as --{key}={val}
        The logs bufferized. They will be printed if the job fails, or sent as output of the function
        Errors are provided with the internal stderr.
        """
        full_command = (
            self.command + [str(x) for x in args] + [f"--{x}={y}" for x, y in kwargs.items()]
        )  # TODO bad parsing
        if self.verbose:
            print(f"The following command is sent: \"{' '.join(full_command)}\"")
        if self.ret_stdout:
            with subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=self.cwd,
                env=self.env,
            ) as process:
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()

                try:
                    copy_process_streams(process, stdout_buffer, stderr_buffer, self.verbose)
                except Exception as e:
                    process.kill()
                    process.wait()
                    raise OSError("Job got killed for an unknown reason.") from e
                stdout = stdout_buffer.getvalue().strip()
                stderr = stderr_buffer.getvalue().strip()
                retcode = process.wait()
                if stderr and (retcode and not self.verbose):
                    # We don't print is self.verbose, as it already happened before.
                    print(stderr, file=sys.stderr)
                if retcode:
                    subprocess_error = subprocess.CalledProcessError(
                        retcode, process.args, output=stdout, stderr=stderr
                    )
                    raise OSError(stderr) from subprocess_error
            return stdout

        subprocess.Popen(
            full_command,
            shell=False,
            cwd=self.cwd,
            env=self.env,
        ).wait()
        return ""


# pylint: disable=too-many-locals
def copy_process_streams(
    process: subprocess.Popen, stdout: io.StringIO, stderr: io.StringIO, verbose: bool = False
):
    """
    Reads the given process stdout/stderr and write them to StringIO objects.
    Make sure that there is no deadlock because of pipe congestion.
    If `verbose` the process stdout/stderr are also copying to the interpreter stdout/stderr.
    """

    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())
    poller = select.poll()
    for fd in stream_by_fd:
        poller.register(fd, select.POLLIN | select.POLLPRI)
    while fds:
        # `poll` syscall will wait until one of the registered file descriptors has content.
        ready = poller.poll()
        for fd, _ in ready:
            p_stream, string, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2**16)
            if not raw_buf:
                fds.remove(fd)
                poller.unregister(fd)
                continue
            buf = raw_buf.decode()
            string.write(buf)
            string.flush()
            if verbose:
                std.write(buf)
                std.flush()