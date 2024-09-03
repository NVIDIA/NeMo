import inspect
import os
import shutil
from pathlib import Path, PosixPath, PurePath, WindowsPath
from typing import Generic, Optional, Tuple, TypeVar
import multiprocessing
from contextlib import contextmanager
import time
import random
import tempfile
import logging

from nemo.utils import logging
import pytorch_lightning as pl

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == 'nt':
    BasePath = WindowsPath
else:
    BasePath = PosixPath


SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")


class Connector(BasePath, Generic[SourceT, TargetT]):
    """
    A generic connector class that provides a framework for transforming a source type (SourceT)
    to a target type (TargetT) while handling file paths based on the operating system.

    Attributes
    ----------
        default_path (Optional[Path]): A default path used when no path is explicitly provided.

    Methods
    -------
        init() -> TargetT:
            Should be implemented to initialize the target type from the source type.

        apply(output_path: Path) -> Path:
            Should be implemented to apply the transformation and save the result at the output path.

        __new__(cls, *args, **kwargs) -> 'Connector':
            Creates a new instance of the connector, using default_path if no path is provided.

        __call__(output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
            Processes the transformation and handles file operations like overwriting.

        local_path(base_path: Optional[Path] = None) -> Path:
            Computes the local path for storage based on a base path or a default cache home.

        is_in_cache(base_path: Optional[Path] = None) -> bool:
            Checks if the transformed data is already cached at the specified base path.
    """

    default_path = None
    _process_locks = {}
    _lock_timeout = 60  # 60 seconds
    _max_retries = 5

    @contextmanager
    def _lock_context(self, path: Path):
        lock_file = path.with_suffix(path.suffix + '.lock')
        process_id = os.getpid()
        parent_process_id = os.getppid()

        logging.info(f"Process {process_id} attempting to acquire lock for {path}")

        for attempt in range(self._max_retries):
            try:
                # Check if the lock is held by the parent process or the current process
                if lock_file.exists():
                    with open(lock_file, 'r') as f:
                        lock_holder = int(f.read().strip())
                    if lock_holder in (parent_process_id, process_id):
                        logging.info(f"Process {process_id} proceeding: lock held by parent {parent_process_id} or self")
                        yield
                        return
                    else:
                        logging.info(f"Process {process_id} found existing lock held by {lock_holder}")

                # Acquire in-memory lock for this process
                if process_id not in self._process_locks:
                    self._process_locks[process_id] = multiprocessing.Lock()
                
                with self._process_locks[process_id]:
                    # Ensure the directory exists
                    lock_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Try to create the lock file
                    try:
                        with open(lock_file, 'x') as f:
                            f.write(str(process_id))
                        logging.info(f"Process {process_id} acquired lock for {path}")
                        yield
                        return
                    except FileExistsError:
                        # Double-check if the existing lock is held by parent or self
                        with open(lock_file, 'r') as f:
                            lock_holder = int(f.read().strip())
                        if lock_holder in (parent_process_id, process_id):
                            logging.info(f"Process {process_id} proceeding: lock held by parent {parent_process_id} or self")
                            yield
                            return
                        logging.info(f"Process {process_id} failed to acquire lock, file already exists")

                # Lock file exists and is held by another process, wait and retry
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Process {process_id} waiting {wait_time:.2f} seconds before retry {attempt + 1}/{self._max_retries}")
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Process {process_id} encountered error during lock acquisition: {e}")
                if attempt == self._max_retries - 1:
                    raise

        logging.error(f"Process {process_id} failed to acquire lock for {path} after {self._max_retries} attempts")
        raise TimeoutError(f"Failed to acquire lock for {path} after {self._max_retries} attempts")

    def init(self) -> TargetT:
        raise NotImplementedError()

    def apply(self, output_path: Path) -> Path:
        raise NotImplementedError()

    def __new__(cls, *args, **kwargs):
        if cls.default_path is not None and not args and 'path' not in kwargs:
            # If default_path is set and no arguments are provided, use default_path as the argument
            return super().__new__(cls, cls.default_path)

        return super().__new__(cls, *args, **kwargs)

    def __call__(self, output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
        _output_path = output_path or self.local_path()

        try:
            with self._lock_context(_output_path):
                if overwrite and _output_path.exists():
                    logging.info(f"Removing existing path {_output_path} (overwrite=True)")
                    self._safe_remove(_output_path)

                if not _output_path.exists():
                    logging.info(f"Applying connector to {_output_path}")
                    to_return = self.apply(_output_path)
                    _output_path = to_return or _output_path

        except TimeoutError:
            logging.error(f"Timeout occurred while trying to acquire the lock for {_output_path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while processing {_output_path}: {e}")
            raise
        finally:
            # Always try to remove the lock file
            self._remove_lock_file(_output_path)

        return _output_path

    def _safe_remove(self, path: Path):
        try:
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as e:
            logging.error(f"Error removing {path}: {e}")

    def _remove_lock_file(self, path: Path):
        lock_file = path.with_suffix(path.suffix + '.lock')
        try:
            if lock_file.exists():
                os.remove(lock_file)
                logging.info(f"Removed lock file {lock_file}")
        except Exception as e:
            logging.warning(f"Failed to remove lock file {lock_file}: {e}")

    @classmethod
    def cleanup_stale_locks(cls, base_path: Path):
        """Remove any stale lock files in the given directory."""
        for lock_file in base_path.glob("*.lock"):
            try:
                os.remove(lock_file)
                logging.info(f"Removed stale lock file: {lock_file}")
            except OSError as e:
                logging.warning(f"Failed to remove stale lock file {lock_file}: {e}")

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = base_path
        else:
            from nemo.lightning.base import NEMO_CACHE_HOME

            _base = Path(NEMO_CACHE_HOME)

        return _base / str(self).replace("://", "/")

    def is_in_cache(self, base_path: Optional[Path] = None) -> bool:
        return self.local_path(base_path=base_path).exists()


class ModelConnector(Connector, Generic[SourceT, TargetT]):
    """
    A specialized connector that extends the generic Connector to handle model-specific operations
    such as setup, save, and load using the Lightning framework.

    Methods
    -------
        nemo_setup(model: pl.LightningModule, trainer: Optional[pl.Trainer] = None) -> pl.Trainer:
            Sets up the model and trainer using a specified strategy, preparing it for training or inference.

        nemo_save(output_path: Path, trainer: pl.Trainer):
            Saves the model's state to the specified path using the trainer's current strategy.

        nemo_load(path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True) -> Tuple[Any, pl.Trainer]:
            Loads a model from the specified path, optionally using a CPU-focused strategy, and returns the model and trainer.
    """

    def nemo_setup(self, model: pl.LightningModule, trainer: Optional[pl.Trainer] = None) -> pl.Trainer:
        """
        Sets up the model and trainer using a specified strategy, preparing it for training or inference.

        Args:
            model (pl.LightningModule): The model to be set up.
            trainer (Optional[pl.Trainer]): The trainer to be used, if not provided a new one will be created.

        Returns
        -------
            pl.Trainer: The trainer configured with the model and strategy.
        """
        from nemo.lightning import MegatronStrategy, Trainer
        from nemo.lightning._strategy_lib import megatron_lazy_init_context

        _trainer = trainer or Trainer(
            devices=1, 
            accelerator="cpu", 
            strategy=MegatronStrategy(
                store_optimizer_states=False,
                ckpt_parallel_save=False,
                ckpt_async_save=False
            )
        )

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()

        if not model.state_dict():
            _trainer.strategy.lazy_init = True
            with _trainer.init_module(), megatron_lazy_init_context(model.config):
                model.configure_model()

        return _trainer

    def nemo_save(self, output_path: Path, trainer: pl.Trainer, dump_io: bool = True) -> None:
        """
        Saves the model's state to the specified path using the trainer's current strategy.

        Args:
            output_path (Path): The path where the model checkpoint will be saved.
            trainer (pl.Trainer): The trainer with the strategy to save the model.
            dump_io (bool): If True, the IO configuration will be saved to the output path.
        """
        logging.info(f"Attempting to save model to {output_path}")
        try:
            with self._lock_context(output_path):
                logging.info("Setting up trainer strategy")
                trainer.strategy._setup_optimizers = False
                trainer.strategy._init_model_parallel = False
                trainer.strategy.setup(trainer)
                
                # Use a temporary directory for saving
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / "temp_checkpoint"
                    logging.info(f"Saving checkpoint to temporary path {temp_path}")
                    trainer.save_checkpoint(temp_path)
                    
                    # Move the temporary checkpoint to the final location
                    logging.info(f"Moving checkpoint from {temp_path} to {output_path}")
                    shutil.move(str(temp_path), str(output_path))

                from nemo.lightning.io.pl import TrainerContext
                from nemo.utils.get_rank import is_global_rank_zero

                if is_global_rank_zero() and dump_io:
                    logging.info("Dumping IO configuration")
                    TrainerContext.from_trainer(trainer).io_dump(output_path)
                
                logging.info(f"Model successfully saved to {output_path}")
        except Exception as e:
            logging.error(f"Error occurred while saving model to {output_path}: {e}")
            raise
        finally:
            # Always try to remove the lock file
            self._remove_lock_file(output_path)

    def nemo_load(
        self, path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True
    ) -> Tuple[pl.LightningModule, pl.Trainer]:
        """
        Loads a model from the specified path.

        Args:
            path (Path): The path from which the model will be loaded.
            trainer (Optional[pl.Trainer]): The trainer to be used, if not provided a new one will be created.
            cpu (bool): If True, the model will be loaded with a CPU-focused strategy.

        Returns
        -------
            Tuple[pl.LightningModule, pl.Trainer]: The loaded model and the trainer configured with the model.
        """
        from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib
        from nemo.lightning.io.api import load_context

        model = load_context(path).model
        _trainer = trainer or Trainer(
            devices=1, 
            accelerator="cpu" if cpu else "gpu", 
            strategy=MegatronStrategy()
        )

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()
        # TODO: Fix cpu initialization
        if not model.state_dict():
            if cpu:
                # TODO: Make this more generic
                with _strategy_lib.megatron_cpu_init_context(model.config):
                    model.configure_model()
            else:
                model.configure_model()

        _trainer.strategy.setup(_trainer)
        _trainer.strategy.load_checkpoint(path)

        return model, _trainer

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = base_path
        else:
            from nemo.lightning.base import NEMO_MODELS_CACHE

            _base = Path(NEMO_MODELS_CACHE)

        # If the useu supplied `hf:///path/to/downloaded/my-model/`
        # then extract the last dir-name (i.e. my-model) and append it to _base
        if str(self).startswith('/'):
            return _base / PurePath((str(self))).name
        return _base / str(self).replace("://", "/")

    def on_import_ckpt(self, model: pl.LightningModule):
        if hasattr(self, "tokenizer"):
            model.tokenizer = self.tokenizer
            if hasattr(model, "__io__"):
                model.__io__.tokenizer = self.tokenizer
