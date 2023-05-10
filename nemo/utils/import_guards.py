# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

from nemo.utils import logging


@contextmanager
def optional_import_guard(warn_on_error=False):
    """
    Context manager to wrap optional import.
    Suppresses `ImportError` (also, `ModuleNotFoundError`), adds warning if `warn_on_error` is True.
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
