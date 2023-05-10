from contextlib import contextmanager

from nemo.utils import logging


@contextmanager
def optional_import_guard(warn_on_error=False):
    """
    Context manager to wrap optional import.
    Suppresses ImportError(also, ModuleNotFoundError), adds warning if `warn_on_error` is True.
    Use separately for each library.

    >>> with optional_import_guard():
    ...     import optional_library

    :param warn_on_error: log warning if import resulted in error
    """
    try:
        yield
    except ImportError as e:
        if warn_on_error:
            logging.warning(e)
    finally:
        pass
