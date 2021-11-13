# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# https://docs.gunicorn.org/en/stable/settings.html

# NOTE: Do not import nemo / torch code here
# Gunicorn creates forked processes - and CUDA cannot be used in forked multiprocessing environment.

import shutil

# General config
bind = "0.0.0.0:8000"
workers = 2

# Worker specific config
worker_connections = 1000
timeout = 180  # 3 minutes of timeout


def on_exit(server):
    # delete tmp dir
    print("Shutting down server ...")
    print("Deleteing tmp directory ...")
    shutil.rmtree('tmp/', ignore_errors=True)
    print("Tmp directory deleted !")
