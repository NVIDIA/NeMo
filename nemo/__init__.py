# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import sys

from nemo.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)


class FakeModule(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        # from module import XYZ
        # XYZ will always None
        return None

    def __bool__(self):
        # fake_module will always return False
        return False


class ImportChecker(object):
    def __init__(self, *args):
        self.module_names = set(args)
        self.path_finder = importlib.machinery.PathFinder()

    def find_module(self, fullname, path=None):
        if fullname not in self.module_names:
            return None
        spec = self.path_finder.find_spec(fullname, path)
        if spec:
            # found the package, let the other loader to handle it
            return None
        if fullname in self.module_names:
            return self
        return None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        # add warning that the module is not found here
        module = FakeModule(name)
        sys.modules[name] = module
        return module


# automatically handle "apex" import
sys.meta_path = [ImportChecker('apex')] + sys.meta_path
