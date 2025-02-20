# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ApiEndpoint(BaseModel):
    """
    Represents evaluation Standard API target.api_endpoint object
    """

    url: str = Field(description="Url of the model", default="http://0.0.0.0:8000/v1/completions/")
    model_id: str = Field(description="Name of the model in API", default="triton_model")
    type: str = Field(description="The type of the target", default="completions")


class EvaluationTarget(BaseModel):
    """
    Represents evaluation Standard API target object
    """

    api_endpoint: ApiEndpoint = Field(description="Api endpoint to be used for evaluation")


class ConfigParams(BaseModel):
    """
    Represents evaluation Standard API config.params object
    """

    top_p: float = Field(
        description="Limits to the top tokens within a certain probability",
        default=0.9999999,
    )
    temperature: float = Field(
        description="Temp of 0 indicates greedy decoding, where the token with highest prob is chosen",
        default=0.0000001,
    )
    limit_samples: Optional[Union[int, float]] = Field(
        description="Limit evaluation to `limit` samples. Default: use all samples", default=None
    )
    max_new_tokens: Optional[int] = Field(description="max tokens to generate", default=256)
    max_retries: Optional[int] = Field(description="Number of REST request retries", default=None)
    parallelism: Optional[int] = Field(description="Parallelism to be used", default=None)
    task: Optional[str] = Field(description="Name of the task", default=None)
    timeout: Optional[int] = Field(description="REST response timeout", default=None)
    extra: Optional[Dict[str, Any]] = Field(
        description="Framework specific parameters to be used for evaluation", default_factory=dict
    )


class EvaluationConfig(BaseModel):
    """
    Represents evaluation Standard API config object
    """

    output_dir: str = Field(description="Directory to output the results", default="results")
    supported_endpoint_types: Optional[list[str]] = Field(
        description="Supported endpoint types like chat or completions", default=None
    )
    type: str = Field(description="Name/type of the task")
    params: ConfigParams = Field(description="Parameters to be used for evaluation", default=ConfigParams())


class MisconfigurationError(Exception):
    pass
