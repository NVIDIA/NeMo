# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import typing
from pytriton.model_config import Tensor
import numpy as np

def typedict2tensor(
    typedict_class,
    overwrite_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    defaults: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    def _map_type(type_):
        if type_ is int:
            return np.int32
        elif type_ is float:
            return np.float32
        elif type_ is bool:
            return np.bool_
        elif type_ is str:
            return bytes
        else:
            raise PyTritonBadParameterError(f"Unknown type {type_}")

    def _get_tensor_params(type_):
        count = 0
        while typing.get_origin(type_) is list:
            type_ = typing.get_args(type_)[0]
            count += 1
        count -= 1  # we don't want to count the last dimension
        shape = (-1,) * count if count > 1 else (1,)
        return {"shape": shape, "dtype": _map_type(type_)}

    overwrite_kwargs = overwrite_kwargs or {}
    return tuple(
        Tensor(name=name, **_get_tensor_params(type_), **overwrite_kwargs)
        for name, type_ in typing.get_type_hints(typedict_class).items()
    )