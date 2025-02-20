import pytest
from pathlib import Path, PosixPath, WindowsPath
import sys
import threading

from nemo.lightning.io.deserialize import (
    SafeDeserialization,
    SafePyrefPolicy,
    DeserializationError,
)


class TestSafePyrefPolicy:
    """Tests for the SafePyrefPolicy class."""

    def test_trusted_modules(self):
        """Test that only trusted modules are allowed when trust_remote_code=False."""
        policy = SafePyrefPolicy(trust_remote_code=False)
        
        # Should allow trusted modules and their submodules
        assert policy.allows_import("nemo", "Model")  # Base module
        assert policy.allows_import("nemo.core", "Model")  # Submodule
        assert policy.allows_import("torch", "nn")  # Another trusted module
        assert policy.allows_import("torch.nn", "Module")  # Submodule
        
        # Should block untrusted modules
        assert not policy.allows_import("os", "system")
        assert not policy.allows_import("subprocess", "run")
        assert not policy.allows_import("untrusted_module", "function")

    def test_blocked_modules(self):
        """Test that blocked modules are always blocked, even in unsafe mode."""
        policy = SafePyrefPolicy(trust_remote_code=True)
        
        # System and OS operations
        assert not policy.allows_import("os", "system")
        assert not policy.allows_import("os.path", "exists")
        assert not policy.allows_import("sys", "exit")
        assert not policy.allows_import("subprocess", "run")
        assert not policy.allows_import("subprocess", "Popen")
        assert not policy.allows_import("shutil", "rmtree")
        
        # Serialization and code execution
        assert not policy.allows_import("pickle", "loads")
        assert not policy.allows_import("marshal", "loads")
        assert not policy.allows_import("shelve", "open")
        assert not policy.allows_import("code", "InteractiveInterpreter")
        assert not policy.allows_import("codeop", "compile_command")
        
        # File and I/O operations
        assert not policy.allows_import("io", "open")
        assert not policy.allows_import("tempfile", "NamedTemporaryFile")
        assert not policy.allows_import("pathlib", "Path")
        assert not policy.allows_import("zipfile", "ZipFile")
        assert not policy.allows_import("tarfile", "TarFile")
        
        # Network and IPC
        assert not policy.allows_import("socket", "socket")
        assert not policy.allows_import("asyncio", "create_task")
        assert not policy.allows_import("multiprocessing", "Process")
        assert not policy.allows_import("threading", "Thread")
        
        # System information and configuration
        assert not policy.allows_import("platform", "system")
        assert not policy.allows_import("pwd", "getpwnam")
        assert not policy.allows_import("grp", "getgrnam")
        assert not policy.allows_import("resource", "getrlimit")
        
        # Package management and imports
        assert not policy.allows_import("importlib", "import_module")
        assert not policy.allows_import("pkg_resources", "require")
        assert not policy.allows_import("setuptools", "setup")
        assert not policy.allows_import("distutils", "core")
        
        # Low-level system access
        assert not policy.allows_import("ctypes", "CDLL")
        assert not policy.allows_import("mmap", "mmap")
        assert not policy.allows_import("fcntl", "fcntl")
        assert not policy.allows_import("signal", "signal")
        
        # Debug and development
        assert not policy.allows_import("pdb", "set_trace")
        assert not policy.allows_import("trace", "Trace")
        assert not policy.allows_import("gc", "collect")
        
        # XML processing (potential security risks)
        assert not policy.allows_import("xml.etree.ElementTree", "parse")
        assert not policy.allows_import("xml.sax", "parse")
        assert not policy.allows_import("xml.dom", "parseString")
        
        # Encoding and crypto (potential security risks)
        assert not policy.allows_import("base64", "b64decode")
        assert not policy.allows_import("codecs", "encode")
        assert not policy.allows_import("crypt", "crypt")
        
        # Terminal and process control
        assert not policy.allows_import("pty", "spawn")
        assert not policy.allows_import("tty", "setraw")
        assert not policy.allows_import("termios", "tcgetattr")
        assert not policy.allows_import("pipes", "Template")
        
        # System logging
        assert not policy.allows_import("syslog", "syslog")
        
        # Web-related (potential security risks)
        assert not policy.allows_import("wsgiref", "simple_server")
        assert not policy.allows_import("http.server", "HTTPServer")
        assert not policy.allows_import("urllib.request", "urlopen")
        
        # Module and class inspection
        assert not policy.allows_import("inspect", "getsource")
        assert not policy.allows_import("dis", "dis")
        assert not policy.allows_import("ast", "parse")

    def test_dangerous_builtins(self):
        """Test that dangerous builtins are always blocked."""
        policy = SafePyrefPolicy(trust_remote_code=True)
        
        # Should block dangerous builtins even in unsafe mode
        assert not policy.allows_import("builtins", "eval")
        assert not policy.allows_import("builtins", "exec")
        assert not policy.allows_import("builtins", "__import__")

    def test_name_validation(self):
        """Test validation of module and symbol names."""
        policy = SafePyrefPolicy()
        
        # Valid names
        assert policy._validate_name("valid_name")
        assert policy._validate_name("ValidName123")
        
        # Invalid names
        assert not policy._validate_name("_hidden")
        assert not policy._validate_name("1invalid")
        assert not policy._validate_name("invalid-name")
        assert not policy._validate_name("../path")


class TestSafeDeserialization:
    """Tests for the SafeDeserialization class."""

    def test_basic_deserialization(self):
        """Test basic deserialization of safe types."""
        safe_data = {
            "root": {
                "type": "leaf",
                "value": {
                    "key": "value",
                    "number": 42,
                    "boolean": True
                }
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        result = SafeDeserialization(safe_data).result
        assert result == {"key": "value", "number": 42, "boolean": True}

    def test_blocked_imports(self):
        """Test that dangerous imports are blocked."""
        dangerous_data = {
            "root": {
                "type": {
                    "type": "pyref",
                    "module": "os",
                    "name": "system"
                },
                "items": [],
                "metadata": None
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError) as exc_info:
            SafeDeserialization(dangerous_data).result
        
        # Verify that the error message indicates a security violation
        assert "not permitted by the active Python reference policy" in str(exc_info.value)

    def test_resource_limits(self):
        """Test resource limits during deserialization."""
        # Create large nested structure
        large_data = {
            "root": {
                "type": "leaf",
                "value": "x" * 1_000_000
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError) as exc_info:
            SafeDeserialization(
                large_data,
                max_string_length=100_000
            ).result
        
        # Verify that the error message indicates a string length violation
        assert "String length" in str(exc_info.value)
        assert "exceeds limit" in str(exc_info.value)


class TestSecurityFeatures:
    """Tests for various security features."""

    def test_dangerous_special_methods(self):
        """Test blocking of objects with dangerous special methods."""
        class DangerousClass:
            def __init__(self):
                pass
                
            def __call__(self):
                return "Dangerous"

        dangerous_obj = DangerousClass()
        policy = SafePyrefPolicy(trust_remote_code=False)
        
        assert not policy.allows_value(dangerous_obj)

    def test_callable_objects(self):
        """Test blocking of callable objects."""
        def dangerous_function():
            return "Dangerous"

        policy = SafePyrefPolicy(trust_remote_code=False)
        assert not policy.allows_value(dangerous_function)

    def test_module_references(self):
        """Test blocking of objects with references to untrusted modules."""
        class ModuleReferenceClass:
            def __init__(self):
                self.__module__ = "os"

        obj = ModuleReferenceClass()
        policy = SafePyrefPolicy(trust_remote_code=False)
        
        assert not policy.allows_value(obj)

    def test_suspicious_strings(self):
        """Test detection of suspicious strings."""
        policy = SafePyrefPolicy()
        
        # Should block strings with suspicious patterns
        suspicious_strings = [
            "__dangerous__",
            "\\x41\\x42\\x43",  # Hex escape
            "\\141\\142\\143",  # Octal escape
            "\\u0041\\u0042",   # Unicode escape
        ]
        
        for s in suspicious_strings:
            assert not policy.allows_value(s)

    def test_nested_dangerous_objects(self):
        """Test detection of dangerous objects nested in safe containers."""
        dangerous_cases = [
            # Lambda in list
            {
                "root": {
                    "type": "leaf",
                    "value": {
                        "safe_key": [1, 2, {"unsafe": lambda x: x}]
                    }
                },
                "objects": {},
                "refcounts": {},
                "version": "0.0.1"
            },
            # Function in dict
            {
                "root": {
                    "type": "leaf",
                    "value": {
                        "unsafe": (lambda: None)
                    }
                },
                "objects": {},
                "refcounts": {},
                "version": "0.0.1"
            },
            # Method in nested structure
            {
                "root": {
                    "type": "leaf",
                    "value": [{"nested": {"deep": {"unsafe": str.strip}}}]
                },
                "objects": {},
                "refcounts": {},
                "version": "0.0.1"
            }
        ]
        
        for case in dangerous_cases:
            with pytest.raises(DeserializationError) as exc_info:
                SafeDeserialization(case).result
            assert "Callable objects are not allowed" in str(exc_info.value)

    def test_type_confusion(self):
        """Test prevention of type confusion attacks."""
        confusing_data = {
            "root": {
                "type": "leaf",
                "value": type("DynamicType", (), {"__call__": lambda self: None})
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError):
            SafeDeserialization(confusing_data).result

    def test_non_string_dict_keys(self):
        """Test that dictionary keys must be strings."""
        invalid_data = {
            "root": {
                "type": "leaf",
                "value": {
                    1: "value"  # numeric key
                }
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError) as exc_info:
            SafeDeserialization(invalid_data).result
        assert "Dictionary keys must be strings" in str(exc_info.value)


class TestErrorHandling:
    """Tests for error handling and reporting."""

    def test_deserialization_error(self):
        """Test proper error handling during deserialization."""
        invalid_data = {
            "root": {
                "type": "invalid_type",
                "items": [],
                "metadata": None
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError):
            SafeDeserialization(invalid_data).result

    def test_security_violation_error(self):
        """Test security violation error handling."""
        policy = SafePyrefPolicy(trust_remote_code=False)
        
        # Should return False for dangerous imports
        assert not policy.allows_import("os", "system")
        
        # The SecurityViolationError should be raised during actual deserialization
        dangerous_data = {
            "root": {
                "type": {
                    "type": "pyref",
                    "module": "os",
                    "name": "system"
                },
                "items": [],
                "metadata": None
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        with pytest.raises(DeserializationError) as exc_info:
            SafeDeserialization(dangerous_data).result
        
        # Verify that the error message indicates a security violation
        assert "not permitted" in str(exc_info.value).lower()

    def test_resource_limit_error(self):
        """Test resource limit error handling."""
        policy = SafePyrefPolicy(max_collection_size=5)
        
        with pytest.raises(ValueError):
            policy._check_collection_size([1, 2, 3, 4, 5, 6])


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_deserialization(self):
        """Test concurrent deserialization operations."""
        safe_data = {
            "root": {"type": "leaf", "value": 42},
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        results = []
        errors = []
        
        def deserialize():
            try:
                result = SafeDeserialization(safe_data).result
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=deserialize) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(r == 42 for r in results)
        assert len(errors) == 0 


class TestPathSerialization:
    """Tests for Path object serialization and deserialization."""

    def test_path_platform_specific(self):
        """Test that Path objects work correctly on different platforms."""
        path_data = {
            "root": {
                "type": "leaf",  # Changed to leaf type
                "value": Path("/some/path")  # Direct Path object
            },
            "objects": {},
            "refcounts": {},
            "version": "0.0.1"
        }
        
        result = SafeDeserialization(path_data).result
        assert isinstance(result, Path)
        
        # Check that we get the right platform-specific path type
        if sys.platform == 'win32':
            assert isinstance(result, WindowsPath)
            assert str(result) == "\\some\\path"
        else:
            assert isinstance(result, PosixPath)
            assert str(result) == "/some/path"
