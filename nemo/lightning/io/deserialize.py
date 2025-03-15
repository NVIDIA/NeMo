import builtins
import logging
import re
import sys
import time
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, Final, List, Optional, Union

from fiddle._src import daglish
from fiddle._src.experimental.serialization import _VALUE_KEY, Deserialization, PyrefPolicy, register_node_traverser

# Configure logging
logger = logging.getLogger(__name__)

# Make critical sets immutable using frozenset
BLOCKED_MODULES: Final[frozenset] = frozenset(
    {
        # System and OS operations
        "os",
        "sys",
        "subprocess",
        "shutil",
        # Serialization and code execution
        "pickle",
        "marshal",
        "shelve",
        "code",
        "codeop",
        # File and I/O operations
        "io",
        "tempfile",
        "pathlib",
        "zipfile",
        "tarfile",
        # Network and IPC
        "socket",
        "asyncio",
        "multiprocessing",
        "threading",
        "http",
        "http.server",
        "urllib",
        "urllib.request",
        "wsgiref",
        # System information and configuration
        "platform",
        "pwd",
        "grp",
        "resource",
        # Package management and imports
        "importlib",
        "pkg_resources",
        "setuptools",
        "distutils",
        # Low-level system access
        "ctypes",
        "mmap",
        "fcntl",
        "signal",
        # Debug and development
        "pdb",
        "trace",
        "gc",
        "inspect",
        "dis",
        "ast",
        # XML processing
        "xml",
        "xml.etree",
        "xml.sax",
        "xml.dom",
        # Encoding and crypto
        "base64",
        "codecs",
        "crypt",
        # Terminal and process control
        "pty",
        "tty",
        "termios",
        "pipes",
        # System logging
        "syslog",
        "logging.handlers",
        # Additional dangerous modules
        "commands",
        "_thread",
        "select",
        "readline",
        "spwd",
        "grp",
        "nis",
        "site",
        "winreg",
        "msvcrt",
        "winsound",
        "venv",
        "uuid",
    }
)

DANGEROUS_BUILTINS: Final[frozenset] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "memoryview",
        "classmethod",
        "staticmethod",
        "property",
        "dir",
        "type",
        "object",
        "super",
        "format",
        "frozenset",
        "help",
        "copyright",
        "credits",
        "license",
        "print",
        "repr",
        "ascii",
        "hash",
        "hex",
        "oct",
        "bin",
        "id",
    }
)

TRUSTED_MODULES: Final[frozenset] = frozenset(
    {
        "nemo",
        "nemo_run",
        "nemo_alligner",
        "nemo_curator",
        "torch",
        "pytorch_lightning",
        "lightning",
        "numpy",
        "collections",
        "typing",
        "enum",
        "dataclasses",
        "pathlib",
    }
)

DANGEROUS_SPECIAL_METHODS: Final[frozenset] = frozenset(
    {
        "__call__",
        "__new__",
        "__init__",
        "__del__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__class_getitem__",
        "__get__",
        "__set__",
        "__delete__",
        "__getattribute__",
        "__slots__",
        "__subclasses__",
        "__bases__",
        "__class__",
        "__mro__",
        "__reduce__",
        "__reduce_ex__",
        "__subclasshook__",
        "__init_subclass__",
        "__prepare__",
        "__instancecheck__",
        "__subclasscheck__",
        "__descr_get__",
        "__descr_set__",
        "__descr_delete__",
        "__delete__",
        "__set_name__",
        "__objclass__",
        "__annotations__",
    }
)


class DeserializationError(Exception):
    """Custom exception for deserialization errors."""

    pass


class SecurityViolationError(DeserializationError):
    """Raised when a security violation is detected."""

    pass


def path_flatten(value: Path) -> tuple[tuple, str]:
    """Flatten a Path object into its string representation."""
    return ((), str(value))


def path_unflatten(values: tuple, metadata: str) -> Path:
    """Reconstruct a Path object from its string representation."""
    return Path(metadata)


def path_elements(value: Path) -> tuple:
    """Return an empty tuple since Path has no traversable elements."""
    return ()


# Register the traverser for Path objects
daglish.register_node_traverser(
    Path,
    flatten_fn=path_flatten,
    unflatten_fn=path_unflatten,
    path_elements_fn=path_elements,
)
register_node_traverser(Path, path_flatten, path_unflatten, path_elements)


class SafePyrefPolicy(PyrefPolicy):
    """A security-enhanced version of PyrefPolicy that restricts module imports."""

    # Class constants
    SAFE_NAME_PATTERN: Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    SUSPICIOUS_PATTERNS: Final = tuple(
        [
            re.compile(r"__[^_\W]+__"),  # Magic methods
            re.compile(r"\\x[0-9a-fA-F]{2}"),  # Hex escapes
            re.compile(r"\\[0-7]{1,3}"),  # Octal escapes
            re.compile(r"\\u[0-9a-fA-F]{4}"),  # Unicode escape
        ]
    )
    DANGEROUS_SPECIAL_METHODS: Final[frozenset] = frozenset(
        {
            "__call__",
            "__new__",
            "__init__",
            "__del__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            "__class_getitem__",
            "__get__",
            "__set__",
            "__delete__",
            "__getattribute__",
            "__slots__",
            "__subclasses__",
            "__bases__",
            "__class__",
            "__mro__",
            "__reduce__",
            "__reduce_ex__",
            "__subclasshook__",
            "__init_subclass__",
            "__prepare__",
            "__instancecheck__",
            "__subclasscheck__",
            "__descr_get__",
            "__descr_set__",
            "__descr_delete__",
            "__delete__",
            "__set_name__",
            "__objclass__",
            "__annotations__",
        }
    )

    def __init__(
        self,
        safe_remote_code: bool = False,
        max_depth: int = 100,
        max_string_length: int = 100_000,
        max_collection_size: int = 10_000,
    ):
        """Initialize the policy with security settings."""
        super().__init__()
        # Initialize all attributes at once using __dict__
        self.__dict__.update(
            {
                "safe_remote_code": safe_remote_code,
                "max_depth": max_depth,
                "max_string_length": max_string_length,
                "max_collection_size": max_collection_size,
                "_current_depth": 0,
                "blocked_modules": BLOCKED_MODULES,
                "trusted_modules": TRUSTED_MODULES,
                # Define safe types as a list instead of a set
                "safe_primitives": (int, float, str, bool, type(None)),
                "safe_collections": (list, tuple, dict, set, frozenset),
                "safe_path_types": (Path, PosixPath, WindowsPath),
            }
        )

    def allows_value(self, value: Any) -> bool:
        """Check if a value is allowed to be deserialized."""
        # Special case for Path class and instances
        if value is Path or isinstance(value, (Path, PosixPath, WindowsPath)):
            return True

        # Check for None
        if value is None:
            return True

        # Check collections first to avoid unhashable type error
        if isinstance(value, (list, tuple, set)):
            if len(value) > self.max_collection_size:
                return False
            return all(self.allows_value(item) for item in value)

        if isinstance(value, dict):
            if len(value) > self.max_collection_size:
                return False
            return all(self.allows_value(k) and self.allows_value(v) for k, v in value.items())

        if isinstance(value, str):
            if len(value) > self.max_string_length:
                return False
            return not any(pattern.search(value) for pattern in self.SUSPICIOUS_PATTERNS)

        # For all other types, just return True if it's a primitive type
        return isinstance(value, (int, float, bool, type(None)))

    def _check_depth(self) -> None:
        """Check and increment the recursion depth."""
        current = self._current_depth
        if current > self.max_depth:
            raise ValueError(f"Maximum recursion depth {self.max_depth} exceeded")
        # Use __dict__ to bypass __setattr__
        self.__dict__["_current_depth"] = current + 1

    def _check_timeout(self) -> None:
        """Check if execution time limit has been exceeded."""
        if time.time() - self._start_time > self.execution_timeout:
            raise ResourceLimitError(f"Execution timeout ({self.execution_timeout}s) exceeded")

    def _validate_int(self, value: int) -> bool:
        """Validate integer values."""
        return -sys.maxsize <= value <= sys.maxsize

    def _validate_float(self, value: float) -> bool:
        """Validate float values."""
        return not (value in (float("inf"), float("-inf"), float("nan")))

    def _validate_str(self, value: str) -> bool:
        """Validate string content."""
        if len(value) > self.max_string_length:
            return False

        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.search(value):
                return False

        return True

    def _validate_bytes(self, value: bytes) -> bool:
        """Validate bytes content."""
        return len(value) <= self.max_string_length

    def _validate_sequence(self, value: Union[List, tuple, set]) -> bool:
        """Validate sequence types."""
        return len(value) <= self.max_collection_size

    def _validate_mapping(self, value: Dict) -> bool:
        """Validate mapping types."""
        if len(value) > self.max_collection_size:
            return False

        # Check key types (should only be strings or numbers)
        return all(isinstance(k, (str, int, float)) for k in value.keys())

    def _validate_name(self, name: str) -> bool:
        """Validate that a name follows safe naming conventions."""
        if not name or len(name) > 64:  # Reasonable maximum length
            return False
        return bool(self.SAFE_NAME_PATTERN.match(name))

    def allows_import(self, module: str, symbol: str) -> bool:
        """Check if importing the given symbol from the module is allowed.

        Args:
            module: The module to import from
            symbol: The symbol to import

        Returns:
            bool: Whether the import is allowed
        """
        # Special case for pathlib.Path
        if module == "pathlib" and symbol == "Path":
            return True

        # Get root module (e.g., 'os.path' -> 'os')
        module_root = module.split(".")[0]

        # Block access to dangerous modules
        if module_root in BLOCKED_MODULES:
            return False

        # Check if it's a dangerous builtin
        if module == "builtins" and symbol in DANGEROUS_BUILTINS:
            logger.warning(f"Blocked import of dangerous builtin: {symbol}")
            return False

        # In restricted mode, only allow trusted modules
        if not self.safe_remote_code:
            # Check if the module or any of its parents are in trusted_modules
            module_parts = module.split(".")
            for i in range(len(module_parts)):
                current_module = ".".join(module_parts[: i + 1])
                if current_module in self.trusted_modules:
                    return True

            logger.warning(f"Blocked import from untrusted module in restricted mode: {module}")
            return False

        # Validate names
        if not self._validate_name(module_root) or not self._validate_name(symbol):
            logger.warning(f"Blocked import due to invalid name: {module}.{symbol}")
            return False

        return True

    def __enter__(self):
        """Context manager to track recursion depth."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset recursion depth on exit."""
        self._current_depth -= 1

    def _check_collection_size(self, value: Any) -> None:
        """Check if a collection exceeds size limits.

        Args:
            value: The collection to check

        Raises:
            ValueError: If the collection size exceeds limits
        """
        if isinstance(value, (str, bytes)):
            if len(value) > self.max_string_length:
                raise ValueError(f"String length {len(value)} exceeds limit {self.max_string_length}")
        elif isinstance(value, (list, tuple, set, dict)):
            if len(value) > self.max_collection_size:
                raise ValueError(f"Collection size {len(value)} exceeds limit {self.max_collection_size}")


class SafeDeserialization(Deserialization):
    """A security-enhanced version of Deserialization that restricts module imports."""

    def __init__(
        self,
        serialized_value: Dict[str, Any],
        safe_remote_code: bool = False,
        pyref_policy: Optional[PyrefPolicy] = None,
        max_depth: int = 100,
        max_string_length: int = 100_000,
        max_collection_size: int = 10_000,
    ):
        if not isinstance(serialized_value, dict):
            raise TypeError("serialized_value must be a dictionary")

        # Store limits
        self.max_string_length = max_string_length
        self.max_collection_size = max_collection_size

        # Verify the integrity of our security measures
        self._verify_security_integrity()

        try:
            if pyref_policy is None:
                pyref_policy = SafePyrefPolicy(
                    safe_remote_code=safe_remote_code,
                    max_depth=max_depth,
                    max_string_length=max_string_length,
                    max_collection_size=max_collection_size,
                )
            super().__init__(serialized_value, pyref_policy=pyref_policy)
        except Exception as e:
            logger.error(f"Deserialization failed: {str(e)}")
            raise DeserializationError(f"Failed to deserialize: {str(e)}") from e

    @staticmethod
    def _verify_security_integrity():
        """Verify that security measures haven't been compromised."""
        # Check if critical sets are still frozen
        if not isinstance(DANGEROUS_BUILTINS, frozenset):
            raise SecurityViolationError("Security violation: DANGEROUS_BUILTINS has been modified")
        if not isinstance(BLOCKED_MODULES, frozenset):
            raise SecurityViolationError("Security violation: BLOCKED_MODULES has been modified")
        if not isinstance(TRUSTED_MODULES, frozenset):
            raise SecurityViolationError("Security violation: TRUSTED_MODULES has been modified")

        # Verify SafePyrefPolicy hasn't been tampered with
        if not hasattr(SafePyrefPolicy, "__setattr__"):
            raise SecurityViolationError("Security violation: SafePyrefPolicy protection removed")

        # Verify that built-in functions haven't been monkey-patched
        for func_name in ("getattr", "setattr", "delattr", "eval", "exec"):
            builtin_func = getattr(builtins, func_name)
            builtins_func = (
                __builtins__[func_name] if isinstance(__builtins__, dict) else getattr(__builtins__, func_name)
            )
            if builtin_func is not builtins_func:
                raise SecurityViolationError(f"Security violation: built-in {func_name} has been modified")

    def _validate_value(self, value: Any) -> None:
        """Recursively validate a value for security concerns."""
        # Check for dangerous types
        if callable(value):
            raise DeserializationError(f"Callable objects are not allowed: {value}")

        # Check collections recursively
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    raise DeserializationError(f"Dictionary keys must be strings, got: {type(k)}")
                self._validate_value(v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                self._validate_value(item)

    def _deserialize_leaf(self, leaf):
        """Override leaf deserialization to add security checks."""
        value = leaf[_VALUE_KEY]

        # Check string length limits
        if isinstance(value, str) and len(value) > self.max_string_length:
            raise DeserializationError(f"String length {len(value)} exceeds limit {self.max_string_length}")

        # Validate nested structures
        self._validate_value(value)

        return value

    def __setattr__(self, name, value):
        if hasattr(self, "_initialized"):
            raise AttributeError("Cannot modify SafeDeserialization attributes")
        super().__setattr__(name, value)
        if name == "_result":  # Last attribute set in parent's __init__
            self._initialized = True

    def __delattr__(self, name):
        raise AttributeError("Cannot delete SafeDeserialization attributes")
