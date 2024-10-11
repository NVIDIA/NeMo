import json
import shutil
import threading
from copy import deepcopy
from pathlib import Path
from pydoc import locate
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import fiddle as fdl
from fiddle._src.experimental import serialization
from typing_extensions import Self

from nemo.lightning.io.artifact.base import Artifact
from nemo.lightning.io.connector import ModelConnector
from nemo.lightning.io.fdl_torch import enable as _enable_ext
from nemo.lightning.io.registry import _io_init, _io_register_serialization, _io_transform_args, _io_wrap_init, track_io
from nemo.utils import logging

ConnT = TypeVar("ConnT", bound=ModelConnector)
CkptType = TypeVar("CkptType")
_enable_ext()


# Thread-local storage for artifacts directory
_thread_local = threading.local()


class IOMixin:
    """
    A mixin class designed to capture the arguments passed to the `__init__` method,
    facilitating the re-creation of the object through `io.reinit` method using stored configurations.

    This class intercepts the initialization of an object to store the arguments in a configuration
    object, which can be serialized and later used to reinitialize the object to its original state.
    It utilizes `fdl.Config` from the Fiddle library to create a structured configuration object
    that holds the initialization parameters. This configuration object is crucial for enabling
    serialization and deserialization of the parameters, thus allowing the object to be reconstructed
    at a later time with the same initial state.

    Attributes
    ----------
        __io__ (fdl.Config[Self]): A configuration object that stores the captured initialization
        parameters in a structured format. This object is an instance of `fdl.Config`, which allows
        for the serialization and deserialization of the parameters, enabling the object to be
        reconstructed at a later time with the same initial state.

    Examples
    --------
        from nemo.lightning import io

        class ExampleClass(io.IOMixin):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        # Creating an instance of ExampleClass
        example = ExampleClass('value1', 'value2')
        example_copy = io.reinit(example)


    Note:
        For more information on `fdl.Config`, refer to the Fiddle library documentation at
        [Fiddle Config Documentation](https://fiddle.readthedocs.io/en/latest/api_reference/core.html#config).

    """

    __io__: fdl.Config[Self]

    def __new__(cls, *args, **kwargs):
        """
        Overrides the default object creation process to wrap the `__init__` method, allowing
        initialization arguments to be captured and stored in the `__io__` attribute.

        Args:
            *args: Variable length argument list for the `__init__` method.
            **kwargs: Arbitrary keyword arguments for the `__init__` method.

        Returns
        -------
            The newly created object instance.
        """
        cls = _io_wrap_init(cls)
        output = object().__new__(cls)

        return output

    def __init_subclass__(cls):
        _io_register_serialization(cls)

    def io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
        """
        Transforms and captures the arguments passed to the `__init__` method, filtering out
        any arguments that are instances of `IOProtocol` or are dataclass fields with default
        factories.

        Args:
            init_fn (Callable): The original `__init__` method of the class.
            *args: Variable length argument list for the `__init__` method.
            **kwargs: Arbitrary keyword arguments for the `__init__` method.

        Returns
        -------
            Dict[str, Any]: A dictionary of the captured and transformed arguments.
        """
        return _io_transform_args(self, init_fn, *args, **kwargs)

    def io_init(self, **kwargs) -> fdl.Config[Self]:
        """
        Initializes the configuration object (`__io__`) with the captured arguments.

        Args:
            **kwargs: A dictionary of arguments that were captured during object initialization.

        Returns
        -------
            fdl.Config[Self]: The initialized configuration object.
        """
        return _io_init(self, **kwargs)

    @classmethod
    def io_artifacts(cls) -> List[Artifact]:
        return []

    def io_dump(self, output: Path):
        """
        Serializes the configuration object (`__io__`) to a file, allowing the object state to be
        saved and later restored. Also creates an artifacts directory and stores it in a thread-local
        global variable. If the artifacts directory is empty at the end, it is deleted.

        Args:
            output (Path): The path to the directory where the configuration object and artifacts
                           will be stored.
        """
        output_path = Path(output)
        local_artifacts_dir = "."
        artifacts_dir = output_path / local_artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Store artifacts directory in thread-local storage
        _thread_local.local_artifacts_dir = local_artifacts_dir
        _thread_local.output_path = output_path

        config_path = output_path / "io.json"
        with open(config_path, "w") as f:
            io = deepcopy(self.__io__)
            _artifact_transform_save(io, output_path, local_artifacts_dir)
            json = serialization.dump_json(io)
            f.write(json)

        # Clear thread-local storage after io_dump is complete
        del _thread_local.local_artifacts_dir
        del _thread_local.output_path

        # Check if artifacts directory is empty and delete if so
        if not any(artifacts_dir.iterdir()):
            shutil.rmtree(artifacts_dir)


class ConnectorMixin:
    """
    A mixin class that provides methods to register and retrieve model connectors for importing
    and exporting models. This class supports dynamic registration of connectors based on file
    extensions, which facilitates the customization and extension of model serialization and
    deserialization processes.

    Attributes
    ----------
        _IMPORTERS (Dict[str, Type[ModelConnector]]): A dictionary mapping file extensions to
            model connector classes that handle the import process.
        _EXPORTERS (Dict[str, Type[ModelConnector]]): A dictionary mapping file extensions to
            model connector classes that handle the export process.
    """

    _IMPORTERS: Dict[str, Type[ModelConnector]] = {}
    _EXPORTERS: Dict[str, Type[ModelConnector]] = {}

    @classmethod
    def import_from(cls, path: str) -> Self:
        """
        Creates an instance of a model by using the appropriate importer based on the file
        extension of the provided path.

        Args:
            path (str): The path to the model file to be imported.

        Example:
            from nemo.collections import llm
            model = llm.Mistral7BModel.import_from("hf")

        Returns
        -------
            Self: An instance of the model initialized from the imported data.
        """
        output = cls._get_connector(path).init()
        output.ckpt_path = output.import_ckpt(path)

        return output

    @classmethod
    def register_importer(cls, ext: str, default_path: Optional[str] = None) -> Callable[[Type[ConnT]], Type[ConnT]]:
        """
        A class method decorator to register a model connector as an importer for a specific file
        extension.

        Args:
            ext (str): The file extension to associate with the model connector.
            default_path (Optional[str]): The default path to use if no path is specified during import.

        Returns
        -------
            Callable[[Type[ConnT]], Type[ConnT]]: The decorator that registers the model connector.
        """

        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._IMPORTERS[str(cls) + ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector

        return decorator

    @classmethod
    def register_exporter(cls, ext: str, default_path: Optional[str] = None) -> Callable[[Type[ConnT]], Type[ConnT]]:
        """
        A class method decorator to register a model connector as an exporter for a specific file
        extension.

        Args:
            ext (str): The file extension to associate with the model connector.
            default_path (Optional[str]): The default path to use if no path is specified during export.

        Returns
        -------
            Callable[[Type[ConnT]], Type[ConnT]]: The decorator that registers the model connector.
        """

        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._EXPORTERS[str(cls) + ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector

        return decorator

    @classmethod
    def importer(cls, path: str) -> ModelConnector:
        """
        Retrieves the appropriate model connector for importing based on the extension of the
        provided path.

        Args:
            path (str): The path to the model file to be imported.

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the import.
        """
        return cls._get_connector(path, importer=True)

    @classmethod
    def exporter(cls, ext: str, path: Union[str, Path]) -> ModelConnector:
        """
        Retrieves the appropriate model connector for exporting based on the extension.

        Args:
            ext (str): The file extension associated with the model connector.
            path (Union[str, Path]): The path where the model will be exported.

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the export.
        """
        return cls._get_connector(ext, path, importer=False)

    def import_ckpt(self, path: str, overwrite: bool = False, base_path: Optional[Path] = None, **kwargs) -> Path:
        """
        Imports a checkpoint from a specified path, potentially overwriting existing files.

        Args:
            path (str): The path to the checkpoint file to be imported.
            overwrite (bool): Flag to determine if existing files should be overwritten (default is False).
            base_path (Optional[Path]): The base path where the checkpoint file is located; used to resolve
                                        relative paths.

        Returns
        -------
            Path: The path to the imported checkpoint.

        Raises
        ------
            FileNotFoundError: If the checkpoint file does not exist at the specified path.
        """
        connector = self._get_connector(path, **kwargs)
        ckpt_path: Path = connector.local_path(base_path=base_path)
        ckpt_path = connector(ckpt_path, overwrite=overwrite)
        connector.on_import_ckpt(self)
        return ckpt_path

    @classmethod
    def _get_connector(
        cls, ext: Union[str, Path], path: Optional[Union[str, Path]] = None, importer: bool = True, **kwargs
    ) -> ModelConnector:
        """
        Retrieves the appropriate model connector based on the file extension and path,
        distinguishing between importers and exporters.

        Args:
            ext (Union[str, Path]): The file extension or a URI that may include a protocol specifier.
            path (Optional[Union[str, Path]]): The path where the model file is located or will be saved.
            importer (bool): Flag to determine if the connector is for importing (True) or exporting (False).

        Returns
        -------
            ModelConnector: The model connector instance capable of handling the import or export.

        Raises
        ------
            ValueError: If no connector is found for the specified extension or if no default path is provided
                        when required.
        """
        _path = None
        ext = str(ext)
        if "://" in ext:
            ext, _path = ext.split("://")
        else:
            _path = str(path)

        connector = cls._IMPORTERS.get(str(cls) + ext) if importer else cls._EXPORTERS.get(str(cls) + ext)
        if not connector:
            raise ValueError(f"No connector found for extension '{ext}'")

        if not _path:
            if not connector.default_path:
                raise ValueError(f"No default path specified for extension '{ext}'. ", "Please provide a path")

            return connector()

        return connector(_path, **kwargs)


def _artifact_transform_save(cfg: fdl.Config, output_path: Path, relative_dir: Path = "."):
    for artifact in getattr(cfg.__fn_or_cls__, "__io_artifacts__", []):
        # Allow optional artifacts
        if artifact.skip:
            continue
        current_val = getattr(cfg, artifact.attr)
        if current_val is None:
            if artifact.required:
                raise ValueError(f"Artifact '{artifact.attr}' is required but not provided")
            continue
        ## dump artifact and return the relative path
        new_val = artifact.dump(current_val, output_path, relative_dir)
        setattr(cfg, artifact.attr, new_val)

    for attr in dir(cfg):
        try:
            if isinstance(getattr(cfg, attr), fdl.Config):
                _artifact_transform_save(getattr(cfg, attr), output_path=output_path, relative_dir=relative_dir)
        except ValueError:
            pass


def _artifact_transform_load(cfg: fdl.Config, path: Path):
    for artifact in getattr(cfg.__fn_or_cls__, "__io_artifacts__", []):
        if artifact.skip:
            continue
        current_val = getattr(cfg, artifact.attr)
        # __init__ arguments can be None
        if current_val is None:
            continue
        ## replace local path with absolute one
        new_val = str(Path(path) / current_val)
        setattr(cfg, artifact.attr, new_val)

    for attr in dir(cfg):
        try:
            if isinstance(getattr(cfg, attr), fdl.Config):
                _artifact_transform_load(getattr(cfg, attr), path=path)
        except ValueError:
            pass


def load(path: Path, output_type: Type[CkptType] = Any, subpath: Optional[str] = None) -> CkptType:
    """
    Loads a configuration from a pickle file and constructs an object of the specified type.

    Args:
        path (Path): The path to the pickle file or directory containing 'io.pkl'.
        output_type (Type[CkptType]): The type of the object to be constructed from the loaded data.
        subpath (Optional[str]): Subpath to selectively load only specific objects inside the output_type. Defaults to None.

    Returns
    -------
        CkptType: An instance of the specified type constructed from the loaded configuration.

    Raises
    ------
        FileNotFoundError: If the specified file does not exist.

    Example:
        loaded_model = load("/path/to/model", output_type=MyModel)
    """
    _path = Path(path)
    _thread_local.output_dir = _path

    if hasattr(_path, "is_dir") and _path.is_dir():
        _path = Path(_path) / "io.json"
    elif hasattr(_path, "isdir") and _path.isdir:
        _path = Path(_path) / "io.json"

    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{_path}'")

    if subpath:
        subpath = "<root>." + subpath

    ## add IO functionality to custom objects present in the json file
    with open(_path) as f:
        j = json.load(f)
    for obj, val in j.get("objects", {}).items():
        clss = ".".join([val["type"]["module"], val["type"]["name"]])
        if subpath and "paths" in val:
            if all(map(lambda p: subpath not in p, val["paths"])):
                continue

        if not serialization.find_node_traverser(locate(clss)):
            track_io(locate(clss))

    with open(_path, "rb") as f:
        json_config = json.loads(f.read())

    root_key = None
    for obj, val in json_config.get("objects", {}).items():
        if "paths" in val and subpath in val["paths"]:
            root_key = obj
            break

    if subpath and not root_key:
        logging.warning(f"Could not find {subpath} for {output_type} in {_path}")

    if root_key:
        json_config["root"]["key"] = root_key

    config = serialization.Deserialization(json_config).result
    _artifact_transform_load(config, path)

    return fdl.build(config)
