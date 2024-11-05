import importlib


def is_lib_available(name: str) -> bool:
    try:
        _ = importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


TRITON_AVAILABLE = is_lib_available("triton")
K2_AVAILABLE = is_lib_available("k2")
