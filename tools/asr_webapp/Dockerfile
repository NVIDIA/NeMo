# syntax=docker/dockerfile:experimental

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

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:1.4.0

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# copy asr service source into a temp directory
WORKDIR /tmp/nemo-service
COPY . .

# override nemo installation with dependency from requirements.txt
RUN /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
RUN cd /tmp/nemo-service && pip install -r "requirements.txt"

# copy webapp into container for end user
WORKDIR /workspace/nemo-service
COPY . /workspace/nemo-service/

# pre-import some asr dependencies which would take time on each worker
RUN python -c "import nemo.collections.asr as nemo_asr"

# expose port 8000 and launch gunicorn
EXPOSE 8000
CMD ["gunicorn", "wsgi:app"]
