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


"""Interfaces common to all Neural Modules and Models."""
import copy
import hashlib
import inspect
import os
import shutil
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import hydra
import torch
import wrapt
from huggingface_hub import HfApi, ModelCard, ModelCardData, ModelFilter
from huggingface_hub import get_token as get_hf_token
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import SoftTemporaryDirectory
from omegaconf import DictConfig, OmegaConf

import nemo
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult
from nemo.utils import logging
from nemo.utils.cloud import maybe_download_from_cloud
from nemo.utils.data_utils import resolve_cache_dir
from nemo.utils.model_utils import import_class_by_path, maybe_update_config_version

__all__ = ['Typing', 'FileIO', 'Model', 'Serialization', 'typecheck', 'PretrainedModelInfo']

_TYPECHECK_ENABLED = True
_TYPECHECK_SEMANTIC_CHECK_ENABLED = True
# TODO @blisc: Remove _HAS_HYDRA
_HAS_HYDRA = True


NEMO_DEFAULT_MODEL_CARD_TEMPLATE = """---
{card_data}
---

# {model_name}

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-PUT-YOUR-ARCHITECTURE-HERE-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-PUT-YOUR-MODEL-SIZE-HERE-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-PUT-YOUR-LANGUAGE-HERE-lightgrey#model-badge)](#datasets)

**Put a short model description here.**

See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/index.html) for complete architecture details.


## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest Pytorch version.
```
pip install nemo_toolkit['all']
``` 

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

**NOTE**: Please update the model class below to match the class of the model being uploaded.

```python
import nemo.core import ModelPT
model = ModelPT.from_pretrained("{repo_id}")
```

### NOTE

    Add some information about how to use the model here. An example is provided for ASR inference below.

    ### Transcribing using Python
    First, let's get a sample
    ```
    wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
    ```
    Then simply do:
    ```
    asr_model.transcribe(['2086-149220-0033.wav'])
    ```

    ### Transcribing many audio files

    ```shell
    python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
     pretrained_name="{repo_id}" \
     audio_dir=""
    ```

### Input

**Add some information about what are the inputs to this model**

### Output

**Add some information about what are the outputs of this model**

## Model Architecture

**Add information here discussing architectural details of the model or any comments to users about the model.**

## Training

**Add information here about how the model was trained. It should be as detailed as possible, potentially including the the link to the script used to train as well as the base config used to train the model. If extraneous scripts are used to prepare the components of the model, please include them here.**

### NOTE

    An example is provided below for ASR

    The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml).

    The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).


### Datasets

**Try to provide as detailed a list of datasets as possible. If possible, provide links to the datasets on HF by adding it to the manifest section at the top of the README (marked by ---).**

### NOTE

    An example for the manifest section is provided below for ASR datasets

    datasets:
    - librispeech_asr
    - fisher_corpus
    - Switchboard-1
    - WSJ-0
    - WSJ-1
    - National-Singapore-Corpus-Part-1
    - National-Singapore-Corpus-Part-6
    - vctk
    - voxpopuli
    - europarl
    - multilingual_librispeech
    - mozilla-foundation/common_voice_8_0
    - MLCommons/peoples_speech

    The corresponding text in this section for those datasets is stated below -

    The model was trained on 64K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams.

    The training dataset consists of private subset with 40K hours of English speech plus 24K hours from the following public datasets:

    - Librispeech 960 hours of English speech
    - Fisher Corpus
    - Switchboard-1 Dataset
    - WSJ-0 and WSJ-1
    - National Speech Corpus (Part 1, Part 6)
    - VCTK
    - VoxPopuli (EN)
    - Europarl-ASR (EN)
    - Multilingual Librispeech (MLS EN) - 2,000 hour subset
    - Mozilla Common Voice (v7.0)
    - People's Speech  - 12,000 hour subset


## Performance

**Add information here about the performance of the model. Discuss what is the metric that is being used to evaluate the model and if there are external links explaning the custom metric, please link to it.

### NOTE

    An example is provided below for ASR metrics list that can be added to the top of the README
    
    model-index:
    - name: PUT_MODEL_NAME
      results:
      - task:
          name: Automatic Speech Recognition
          type: automatic-speech-recognition
        dataset:
          name: AMI (Meetings test)
          type: edinburghcstr/ami
          config: ihm
          split: test
          args:
            language: en
        metrics:
        - name: Test WER
          type: wer
          value: 17.10
      - task:
          name: Automatic Speech Recognition
          type: automatic-speech-recognition
        dataset:
          name: Earnings-22
          type: revdotcom/earnings22
          split: test
          args:
            language: en
        metrics:
        - name: Test WER
          type: wer
          value: 14.11

Provide any caveats about the results presented in the top of the discussion so that nuance is not lost. 

It should ideally be in a tabular format (you can use the following website to make your tables in markdown format - https://www.tablesgenerator.com/markdown_tables)**

## Limitations

**Discuss any practical limitations to the model when being used in real world cases. They can also be legal disclaimers, or discussion regarding the safety of the model (particularly in the case of LLMs).**


### Note

    An example is provided below 

    Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. The model might also perform worse for accented speech.


## References

**Provide appropriate references in the markdown link format below. Please order them numerically.**

[1] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
"""


def is_typecheck_enabled():
    """
    Getter method for typechecking state.
    """
    return _TYPECHECK_ENABLED


def is_semantic_typecheck_enabled():
    """
    Getter method for typechecking semantics state.
    """
    return _TYPECHECK_SEMANTIC_CHECK_ENABLED


@dataclass
class TypecheckMetadata:
    """
    Metadata class for input/output neural types.

    # Primary attributes
    original_types: Preserve the dictionary of type information provided.

    ignore_collections: For backward compatibility, container support can be disabled explicitly
        using this flag. When set to True, all nesting is ignored and nest-depth checks are skipped.

    # Derived attributed
    mandatory_types: Sub-dictionary of `original_types` which contains only those types which
        are mandatory to include when calling the function.

    base_types: Dictionary of flattened `str: NeuralType` definitions, disregarding the nest level
        details into appropriate arguments.

    container_depth: Dictionary mapping `str: int` - such that the valid depth of the nest of this
        neural type is recorded.

    has_container_types: Bool flag declaring if any of the neural types declares a container nest
        in its signature.

    is_singular_container_type: Bool flag declaring if this is a single Neural Type with a container
        nest in its signature. Required for supporting python list expansion in return statement.

    """

    original_types: Dict[str, NeuralType]
    ignore_collections: bool

    mandatory_types: Dict[str, NeuralType] = field(init=False)
    base_types: Dict[str, NeuralType] = field(init=False)

    container_depth: Dict[str, int] = field(init=False)
    has_container_types: bool = field(init=False)
    is_singular_container_type: bool = field(init=False)

    def __post_init__(self):
        # If even one NeuralType declares a container nest, set to True
        has_container_types = False
        for type_val in self.original_types.values():
            if isinstance(type_val, (list, tuple)):
                has_container_types = True
                break
        self.has_container_types = has_container_types

        # If only one NeuralType is declared, and it declares a container nest, set to True
        if self.has_container_types and len(self.original_types) == 1:
            self.is_singular_container_type = True
        else:
            self.is_singular_container_type = False

        # If container nests are declared, flatten the nest into `base_types`
        # Also compute the nest depth for each of the NeuralTypes
        if self.has_container_types:
            self.base_types = {}
            self.container_depth = {}

            for type_key, type_val in self.original_types.items():
                depth = 0
                while isinstance(type_val, (list, tuple)):
                    if len(type_val) > 1:
                        raise TypeError(
                            f"Neural Type `{type_key}`: {type_val} definition contains more than one element when"
                            "declaring the nested container structure.\n"
                            "Please ensure that you have only 1 NeuralType inside of the entire nested structure "
                            "definition."
                        )

                    type_val = type_val[0]
                    depth += 1

                self.base_types[type_key] = type_val
                self.container_depth[type_key] = depth
        else:
            # Otherwise, simply preserve the original_types and set depth of nest to 0.
            self.base_types = self.original_types
            self.container_depth = {type_key: 0 for type_key in self.base_types.keys()}

        # Compute subset of original_types which are mandatory in the call argspec
        self.mandatory_types = {
            type_key: type_val for type_key, type_val in self.base_types.items() if not type_val.optional
        }


class Typing(ABC):
    """
    An interface which endows module with neural types
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return None

    def _validate_input_types(self, input_types=None, ignore_collections=False, **kwargs):
        """
        This function does a few things.

        1) It ensures that len(self.input_types <non-optional>) <= len(kwargs) <= len(self.input_types).
        2) For each (keyword name, keyword value) passed as input to the wrapped function:
            - Check if the keyword name exists in the list of valid self.input_types names.
            - Check if keyword value has the `neural_type` property.
                - If it does, then perform a comparative check and assert that neural types
                    are compatible (SAME or GREATER).
            - Check if keyword value is a container type (list or tuple). If yes,
                then perform the elementwise test of neural type above on each element
                of the nested structure, recursively.

        Args:
            input_types: Either the `input_types` defined at class level, or the local function
                overridden type definition.
            ignore_collections: For backward compatibility, container support can be disabled explicitly
                using this flag. When set to True, all nesting is ignored and nest-depth checks are skipped.
            kwargs: Dictionary of argument_name:argument_value pairs passed to the wrapped
                function upon call.
        """
        if input_types is not None:
            # Precompute metadata
            metadata = TypecheckMetadata(original_types=input_types, ignore_collections=ignore_collections)

            total_input_types = len(input_types)
            mandatory_input_types = len(metadata.mandatory_types)

            # Allow number of input arguments to be <= total input neural types.
            if len(kwargs) < mandatory_input_types or len(kwargs) > total_input_types:
                raise TypeError(
                    f"Number of input arguments provided ({len(kwargs)}) is not as expected. Function has "
                    f"{total_input_types} total inputs with {mandatory_input_types} mandatory inputs."
                )

            for key, value in kwargs.items():
                # Check if keys exists in the defined input types
                if key not in input_types:
                    raise TypeError(
                        f"Input argument {key} has no corresponding input_type match. "
                        f"Existing input_types = {input_types.keys()}"
                    )

                # Perform neural type check
                if (
                    hasattr(value, 'neural_type')
                    and is_semantic_typecheck_enabled()
                    and not metadata.base_types[key].compare(value.neural_type)
                    in (NeuralTypeComparisonResult.SAME, NeuralTypeComparisonResult.GREATER,)
                ):
                    error_msg = [
                        f"{input_types[key].compare(value.neural_type)} :",
                        f"Input type expected : {input_types[key]}",
                        f"Input type found : {value.neural_type}",
                        f"Argument: {key}",
                    ]
                    for i, dict_tuple in enumerate(metadata.base_types[key].elements_type.type_parameters.items()):
                        error_msg.insert(i + 2, f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    for i, dict_tuple in enumerate(value.neural_type.elements_type.type_parameters.items()):
                        error_msg.append(f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    raise TypeError("\n".join(error_msg))

                # Perform input ndim check
                if hasattr(value, 'shape'):
                    value_shape = value.shape
                    type_shape = metadata.base_types[key].axes
                    name = key

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Input shape expected = {metadata.base_types[key].axes} | \n"
                            f"Input shape found : {value_shape}"
                        )

                # Perform recursive neural type check for homogeneous elements
                elif isinstance(value, list) or isinstance(value, tuple):
                    for ind, val in enumerate(value):
                        """
                        This initiates a DFS, tracking the depth count as it goes along the nested structure.
                        Initial depth is 1 as we consider the current loop to be the 1st step inside the nest.
                        """
                        self.__check_neural_type(val, metadata, depth=1, name=key)

    def _attach_and_validate_output_types(self, out_objects, ignore_collections=False, output_types=None):
        """
        This function does a few things.

        1) It ensures that len(out_object) == len(self.output_types).
        2) If the output is a tensor (or list/tuple of list/tuple ... of tensors), it
            attaches a neural_type to it. For objects without the neural_type attribute,
            such as python objects (dictionaries and lists, primitive data types, structs),
            no neural_type is attached.

        Note: tensor.neural_type is only checked during _validate_input_types which is
        called prior to forward().

        Args:
            output_types: Either the `output_types` defined at class level, or the local function
                overridden type definition.
            ignore_collections: For backward compatibility, container support can be disabled explicitly
                using this flag. When set to True, all nesting is ignored and nest-depth checks are skipped.
            out_objects: The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if output_types is not None:
            # Precompute metadata
            metadata = TypecheckMetadata(original_types=output_types, ignore_collections=ignore_collections)
            out_types_list = list(metadata.base_types.items())
            mandatory_out_types_list = list(metadata.mandatory_types.items())

            # First convert all outputs to list/tuple format to check correct number of outputs
            if isinstance(out_objects, (list, tuple)):
                out_container = out_objects  # can be any rank nested structure
            else:
                out_container = [out_objects]

            # If this neural type has a *single output*, with *support for nested outputs*,
            # then *do not* perform any check on the number of output items against the number
            # of neural types (in this case, 1).
            # This is done as python will *not* wrap a single returned list into a tuple of length 1,
            # instead opting to keep the list intact. Therefore len(out_container) in such a case
            # is the length of all the elements of that list - each of which has the same corresponding
            # neural type (defined as the singular container type).
            if metadata.is_singular_container_type:
                pass

            # In all other cases, python will wrap multiple outputs into an outer tuple.
            # Allow number of output arguments to be <= total output neural types and >= mandatory outputs.

            elif len(out_container) > len(out_types_list) or len(out_container) < len(mandatory_out_types_list):
                raise TypeError(
                    "Number of output arguments provided ({}) is not as expected. "
                    "It should be larger or equal than {} and less or equal than {}.\n"
                    "This can be either because insufficient/extra number of output NeuralTypes were provided,"
                    "or the provided NeuralTypes {} should enable container support "
                    "(add '[]' to the NeuralType definition)".format(
                        len(out_container), len(out_types_list), len(mandatory_out_types_list), output_types
                    )
                )

            # Attach types recursively, if possible
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                # Here, out_objects is a single object which can potentially be attached with a NeuralType
                try:
                    out_objects.neural_type = out_types_list[0][1]
                except Exception:
                    pass

                # Perform output ndim check
                if hasattr(out_objects, 'shape'):
                    value_shape = out_objects.shape
                    type_shape = out_types_list[0][1].axes
                    name = out_types_list[0][0]

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Output shape expected = {type_shape} | \n"
                            f"Output shape found : {value_shape}"
                        )

            elif metadata.is_singular_container_type:
                # If only a single neural type is provided, and it defines a container nest,
                # then all elements of the returned list/tuple are assumed to belong to that
                # singular neural type.
                # As such, the "current" depth inside the DFS loop is counted as 1,
                # and subsequent nesting will increase this count.

                # NOTE:
                # As the flag `is_singular_container_type` will activate only for
                # the case where there is 1 output type defined with container nesting,
                # this is a safe assumption to make.
                depth = 1

                # NOTE:
                # A user may chose to explicitly wrap the single output list within an explicit tuple
                # In such a case we reduce the "current" depth to 0 - to acknowledge the fact that
                # the actual nest exists within a wrapper tuple.
                if len(out_objects) == 1 and type(out_objects) == tuple:
                    depth = 0

                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, metadata, depth=depth, name=out_types_list[0][0])
            else:
                # If more then one item is returned in a return statement, python will wrap
                # the output with an outer tuple. Therefore there must be a 1:1 correspondence
                # of the output_neural type (with or without nested structure) to the actual output
                # (whether it is a single object or a nested structure of objects).
                # Therefore in such a case, we "start" the DFS at depth 0 - since the recursion is
                # being applied on 1 neural type : 1 output struct (single or nested output).
                # Since we are guarenteed that the outer tuple will be built by python,
                # assuming initial depth of 0 is appropriate.
                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, metadata, depth=0, name=out_types_list[ind][0])

    def __check_neural_type(self, obj, metadata: TypecheckMetadata, depth: int, name: str = None):
        """
        Recursively tests whether the obj satisfies the semantic neural type assertion.
        Can include shape checks if shape information is provided.

        Args:
            obj: Any python object that can be assigned a value.
            metadata: TypecheckMetadata object.
            depth: Current depth of recursion.
            name: Optional name used of the source obj, used when an error occurs.
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__check_neural_type(elem, metadata, depth + 1, name=name)
            return  # after processing nest, return to avoid testing nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While checking input neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        if (
            hasattr(obj, 'neural_type')
            and is_semantic_typecheck_enabled()
            and not type_val.compare(obj.neural_type)
            in (NeuralTypeComparisonResult.SAME, NeuralTypeComparisonResult.GREATER,)
        ):
            raise TypeError(
                f"{type_val.compare(obj.neural_type)} : \n"
                f"Input type expected = {type_val} | \n"
                f"Input type found : {obj.neural_type}"
            )

        # Perform input ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Input shape expected = {type_shape} | \n"
                    f"Input shape found : {value_shape}"
                )

    def __attach_neural_type(self, obj, metadata: TypecheckMetadata, depth: int, name: str = None):
        """
        Recursively attach neural types to a given object - as long as it can be assigned some value.

        Args:
            obj: Any python object that can be assigned a value.
            metadata: TypecheckMetadata object.
            depth: Current depth of recursion.
            name: Optional name used of the source obj, used when an error occurs.
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__attach_neural_type(elem, metadata, depth=depth + 1, name=name)
            return  # after processing nest, return to avoid argument insertion into nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While attaching output neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        try:
            obj.neural_type = type_val
        except Exception:
            pass

        # Perform output ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Output shape expected = {type_shape} | \n"
                    f"Output shape found : {value_shape}"
                )


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: 'DictConfig', trainer: Optional['Trainer'] = None):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if _HAS_HYDRA:
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

            config = maybe_update_config_version(config)

        # Hydra 0.x API
        if ('cls' in config or 'target' in config) and 'params' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        # Hydra 1.x API
        elif '_target_' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            instance = None
            prev_error = ""
            # Attempt class path resolution from config `target` class (if it exists)
            if 'target' in config:
                target_cls = config["target"]  # No guarantee that this is a omegaconf class
                imported_cls = None
                try:
                    # try to import the target class
                    imported_cls = import_class_by_path(target_cls)
                    # if calling class (cls) is subclass of imported class,
                    # use subclass instead
                    if issubclass(cls, imported_cls):
                        imported_cls = cls
                    accepts_trainer = Serialization._inspect_signature_for_trainer(imported_cls)
                    if accepts_trainer:
                        instance = imported_cls(cfg=config, trainer=trainer)
                    else:
                        instance = imported_cls(cfg=config)
                except Exception as e:
                    # record previous error
                    tb = traceback.format_exc()
                    prev_error = f"Model instantiation failed!\nTarget class:\t{target_cls}" f"\nError(s):\t{e}\n{tb}"
                    logging.debug(prev_error + "\nFalling back to `cls`.")

            # target class resolution was unsuccessful, fall back to current `cls`
            if instance is None:
                try:
                    accepts_trainer = Serialization._inspect_signature_for_trainer(cls)
                    if accepts_trainer:
                        instance = cls(cfg=config, trainer=trainer)
                    else:
                        instance = cls(cfg=config)

                except Exception as e:
                    # report saved errors, if any, and raise
                    if prev_error:
                        logging.error(prev_error)
                    raise e

        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> 'DictConfig':
        """Returns object's configuration to config dictionary"""
        if hasattr(self, '_cfg') and self._cfg is not None:
            # Resolve the config dict
            if _HAS_HYDRA and isinstance(self._cfg, DictConfig):
                config = OmegaConf.to_container(self._cfg, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

                config = maybe_update_config_version(config)

            self._cfg = config

            return self._cfg
        else:
            raise NotImplementedError(
                'to_config_dict() can currently only return object._cfg but current object does not have it.'
            )

    @classmethod
    def _inspect_signature_for_trainer(cls, check_cls):
        if hasattr(check_cls, '__init__'):
            signature = inspect.signature(check_cls.__init__)
            if 'trainer' in signature.parameters:
                return True
            else:
                return False
        else:
            return False


class FileIO(ABC):
    def save_to(self, save_path: str):
        """
        Standardized method to save a tarfile containing the checkpoint, config, and any additional artifacts.
        Implemented via :meth:`nemo.core.connectors.save_restore_connector.SaveRestoreConnector.save_to`.

        Args:
            save_path: str, path to where the file should be saved.
        """
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional['Trainer'] = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """
        Restores model instance (weights and configuration) from a .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.
            trainer: An optional Trainer object, passed to the model constructor.
            save_restore_connector: An optional SaveRestoreConnector object that defines the implementation
                of the restore_from() method.
        """
        raise NotImplementedError()

    @classmethod
    def from_config_file(cls, path2yaml_file: str):
        """
        Instantiates an instance of NeMo Model from YAML config file.
        Weights will be initialized randomly.
        Args:
            path2yaml_file: path to yaml file with model configuration

        Returns:

        """
        if issubclass(cls, Serialization):
            conf = OmegaConf.load(path2yaml_file)
            return cls.from_config_dict(config=conf)
        else:
            raise NotImplementedError()

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current instance's configuration to YAML config file. Weights will not be saved.
        Args:
            path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved

        Returns:
        """
        if hasattr(self, '_cfg'):
            self._cfg = maybe_update_config_version(self._cfg)
            with open(path2yaml_file, 'w', encoding='utf-8') as fout:
                OmegaConf.save(config=self._cfg, f=fout, resolve=True)
        else:
            raise NotImplementedError()


@total_ordering
@dataclass
class PretrainedModelInfo:
    pretrained_model_name: str
    description: str
    location: str
    class_: 'Model' = None
    aliases: List[str] = None

    def __repr__(self):
        base = self.__class__.__name__
        extras = (
            "pretrained_model_name={pretrained_model_name},\n\t"
            "description={description},\n\t"
            "location={location}".format(**self.__dict__)
        )

        if self.class_ is not None:
            extras = "{extras},\n\t" "class_={class_}".format(extras=extras, **self.__dict__)

        representation = f"{base}(\n\t{extras}\n)"
        return representation

    def __hash__(self):
        # assumes that locations are unique urls, and therefore their hashes
        # should ideally also be unique
        location_hash = hash(self.location)
        return location_hash

    def __eq__(self, other):
        # another object is equal to self, iff
        # if it's hash is equal to hash(self)
        return hash(self) == hash(other) or self.pretrained_model_name == other.pretrained_model_name

    def __lt__(self, other):
        return self.pretrained_model_name < other.pretrained_model_name


class Model(Typing, Serialization, FileIO):
    """
    Abstract class offering interface which should be implemented by all NeMo models.
    """

    @classmethod
    @abstractmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """
        Should list all pre-trained models available via NVIDIA NGC cloud.
        Note: There is no check that requires model names and aliases to be unique. In the case of a collision, whatever
        model (or alias) is listed first in the this returned list will be instantiated.

        Returns:
            A list of PretrainedModelInfo entries
        """
        pass

    @classmethod
    def search_huggingface_models(
        cls, model_filter: Optional[Union[ModelFilter, List[ModelFilter]]] = None
    ) -> List[ModelInfo]:
        """
        Should list all pre-trained models available via Hugging Face Hub.

        The following metadata can be passed via the `model_filter` for additional results.
        Metadata:
            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
            limit_results: Optional int, limits the number of results returned.

        .. code-block:: python

            # You can replace <DomainSubclass> with any subclass of ModelPT.
            from nemo.core import ModelPT

            # Get default ModelFilter
            filt = <DomainSubclass>.get_hf_model_filter()

            # Make any modifications to the filter as necessary
            filt.language = [...]
            filt.task = ...
            filt.tags = [...]

            # Add any metadata to the filter as needed
            filt.limit_results = 5

            # Obtain model info
            model_infos = <DomainSubclass>.search_huggingface_models(model_filter=filt)

            # Browse through cards and select an appropriate one
            card = model_infos[0]

            # Restore model using `modelId` of the card.
            model = ModelPT.from_pretrained(card.modelId)

        Args:
            model_filter: Optional ModelFilter or List[ModelFilter] (from Hugging Face Hub)
                that filters the returned list of compatible model cards, and selects all results from each filter.
                Users can then use `model_card.modelId` in `from_pretrained()` to restore a NeMo Model.
                If no ModelFilter is provided, uses the classes default filter as defined by `get_hf_model_filter()`.

        Returns:
            A list of ModelInfo entries.
        """
        # Resolve model filter if not provided as argument
        if model_filter is None:
            model_filter = cls.get_hf_model_filter()

        # If single model filter, wrap into list
        if not isinstance(model_filter, Iterable):
            model_filter = [model_filter]

        # Inject `nemo` library filter
        for mfilter in model_filter:
            if isinstance(mfilter.library, str) and mfilter.library != 'nemo':
                logging.warning(f"Model filter's `library` tag updated be `nemo`. Original value: {mfilter.library}")
                mfilter.library = "nemo"

            elif isinstance(mfilter, Iterable) and 'nemo' not in mfilter.library:
                logging.warning(
                    f"Model filter's `library` list updated to include `nemo`. Original value: {mfilter.library}"
                )
                mfilter.library = list(mfilter)
                mfilter.library.append('nemo')

        # Check if api token exists, use if it does
        hf_token = get_hf_token()

        # Search for all valid models after filtering
        api = HfApi()

        # Setup extra arguments for model filtering
        all_results = []  # type: List[ModelInfo]

        for mfilter in model_filter:
            cardData = None
            limit = None

            if hasattr(mfilter, 'resolve_card_info') and mfilter.resolve_card_info is True:
                cardData = True

            if hasattr(mfilter, 'limit_results') and mfilter.limit_results is not None:
                limit = mfilter.limit_results

            results = api.list_models(
                filter=mfilter, token=hf_token, sort="lastModified", direction=-1, cardData=cardData, limit=limit,
            )  # type: Iterable[ModelInfo]

            for result in results:
                all_results.append(result)

        return all_results

    @classmethod
    def get_available_model_names(cls) -> List[str]:
        """
        Returns the list of model names available via NVIDIA NGC cloud,
        to get the complete model description use list_available_models()
        Returns:
            A list of model names
        """
        model_names = []
        if cls.list_available_models() is not None:
            model_names = [model.pretrained_model_name for model in cls.list_available_models()]
        return model_names

    @classmethod
    def get_hf_model_filter(cls) -> ModelFilter:
        """
        Generates a filter for HuggingFace models.

        Additionally includes default values of some metadata about results returned by the Hub.

        Metadata:
            resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
            limit_results: Optional int, limits the number of results returned.

        Returns:
            A Hugging Face Hub ModelFilter object.
        """
        model_filter = ModelFilter(library='nemo')

        # Attach some additional info
        model_filter.resolve_card_info = False
        model_filter.limit_results = None

        return model_filter

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        refresh_cache: bool = False,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional['Trainer'] = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """
        Instantiates an instance of NeMo from NVIDIA NGC cloud
        Use restore_from() to instantiate from a local .nemo file.
        Args:
            model_name: string key which will be used to find the module.
            refresh_cache: If set to True, then when fetching from cloud, this will re-fetch the file
                from cloud even if it is already found in a cache locally.
            override_config_path: path to a yaml config that will override the internal
                config file
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to torch.load_state_dict. By default true.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Returns:
            A model instance of a particular model class or its underlying config (if return_config is set).
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        # Resolve if the pretrained model name is from NGC or other sources
        # HF Hub source
        if '/' in model_name:
            class_, nemo_model_file_in_cache = cls._get_hf_hub_pretrained_model_info(
                model_name=model_name, refresh_cache=refresh_cache
            )

            # Check if nemo_model_file_in_cache is a directory
            if os.path.isdir(nemo_model_file_in_cache):
                # Update SaveRestoreConnector with the flag to read from an unpacked NeMo folder
                save_restore_connector.model_extracted_dir = nemo_model_file_in_cache

        else:
            # NGC source
            class_, nemo_model_file_in_cache = cls._get_ngc_pretrained_model_info(
                model_name=model_name, refresh_cache=refresh_cache
            )

        instance = class_.restore_from(
            restore_path=nemo_model_file_in_cache,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
            return_config=return_config,
            trainer=trainer,
            save_restore_connector=save_restore_connector,
        )
        return instance

    @classmethod
    def _get_ngc_pretrained_model_info(cls, model_name: str, refresh_cache: bool = False) -> Tuple[type, str]:
        """
        Resolve the NGC model pretrained information given a model name.
        Assumes the model subclass implements the `list_available_models()` inherited method.

        Args:
            model_name: Str name of the model. Must be the original name or an alias of the model, without any '/'.
            refresh_cache: Bool, determines whether cache must be refreshed (model is re-downloaded).

        Returns:
            A tuple of details describing :
            -   The resolved class of the model. This requires subclass to implement PretrainedModelInfo.class_.
                If the class cannot be resolved, default to the class that called this method.
            -   The path to the NeMo model (.nemo file) in some cached directory.
        """
        location_in_the_cloud = None
        description = None
        class_ = None
        models = cls.list_available_models()
        if models is not None:
            for pretrained_model_info in cls.list_available_models():
                found = False
                if pretrained_model_info.pretrained_model_name == model_name:
                    found = True
                elif pretrained_model_info.aliases is not None:
                    for alias in pretrained_model_info.aliases:
                        if alias == model_name:
                            found = True
                            break
                if found:
                    location_in_the_cloud = pretrained_model_info.location
                    description = pretrained_model_info.description
                    class_ = pretrained_model_info.class_
                    break

        if location_in_the_cloud is None:
            raise FileNotFoundError(
                f"Model {model_name} was not found. Check cls.list_available_models() for the list of all available models."
            )
        filename = location_in_the_cloud.split("/")[-1]
        url = location_in_the_cloud.replace(filename, "")
        cache_dir = Path.joinpath(resolve_cache_dir(), f'{filename[:-5]}')
        # If either description and location in the cloud changes, this will force re-download
        cache_subfolder = hashlib.md5((location_in_the_cloud + description).encode('utf-8')).hexdigest()
        # if file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True
        nemo_model_file_in_cache = maybe_download_from_cloud(
            url=url, filename=filename, cache_dir=cache_dir, subfolder=cache_subfolder, refresh_cache=refresh_cache
        )

        logging.info("Instantiating model from pre-trained checkpoint")

        if class_ is None:
            class_ = cls

        return class_, nemo_model_file_in_cache

    @classmethod
    def _get_hf_hub_pretrained_model_info(cls, model_name: str, refresh_cache: bool = False) -> Tuple[type, str]:
        """
        Resolve the HuggingFace Hub model pretrained information given a model name.
        The model name must be of general syntax ``{source_repo}/{model_name}``.

        Note:
            The ``{source_repo}`` need not be ``nvidia``, it can be any public repository, even external to Nvidia.
            This allows public, externally contributed models to be run freely using Nvidia NeMo.

        Args:
            model_name: Str name of the model. Must be the original name or an alias of the model, without any '/'.
            refresh_cache: Bool, determines whether cache must be refreshed (model is re-downloaded).

        Returns:
            A tuple of details describing :
            -   The resolved class of the model. Since the source is external to NeMo, always default to using
                the calling class. Depend on target class resolution by restore_from() for calling the correct class.
            -   The path to the NeMo model (.nemo file) in some cached directory (managed by HF Hub).
        """
        # Resolve the model name without origin for filename
        resolved_model_filename = model_name.split("/")[-1] + '.nemo'

        # Check if api token exists, use if it does
        hf_token = get_hf_token()

        # First check if .nemo file exists in HF
        api = HfApi(token=hf_token)

        # Check if model exists in HF
        nemo_file_exists = api.file_exists(repo_id=model_name, filename=resolved_model_filename, repo_type="model")

        if nemo_file_exists:
            # Try to load the model from the Huggingface Hub
            path = hf_hub_download(
                repo_id=model_name,
                filename=resolved_model_filename,
                library_name='nemo',
                library_version=nemo.__version__,
                force_download=refresh_cache,
                token=hf_token,
            )
        else:
            repo_info = api.repo_info(repo_id=model_name, token=hf_token)

            # Download whole HF repo and load entire directory as nemo directory
            cache_dir = Path.joinpath(resolve_cache_dir(), "hf_hub_cache", f'{model_name}')
            # If either description and location in the cloud changes, this will force re-download
            cache_subfolder = repo_info.sha
            # if file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True
            save_path = os.path.join(cache_dir, cache_subfolder)

            if os.path.exists(cache_dir):
                num_files_in_dir = len(os.listdir(cache_dir))
                print("Found {} files in cache directory {}".format(num_files_in_dir, cache_dir))
                print(f"Deleting old cache directory for model `{model_name}` in order to prevent duplicates...")
                shutil.rmtree(cache_dir, ignore_errors=True)

            if not os.path.exists(save_path):
                logging.info(f"Downloading {model_name} from HuggingFace Hub to path: {save_path}")
                os.makedirs(save_path, exist_ok=True)

            path = snapshot_download(
                repo_id=model_name,
                library_name='nemo',
                library_version=nemo.__version__,
                force_download=refresh_cache,
                cache_dir=save_path,
                local_dir=save_path,
                local_dir_use_symlinks=False,
                token=hf_token,
            )

        # Cannot pre-resolve the specific class without double instantiation (first for config, second for model params)
        # Default to current class, and perform basic class path resolution (handled via restore_from() + target class)
        class_ = cls

        return class_, path

    def push_to_hf_hub(
        self,
        repo_id: str,
        *,
        pack_nemo_file: bool = True,
        model_card: Optional[ModelCard] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
    ):
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            pack_nemo_file (`bool`, *optional*, defaults to `True`): Whether to pack the model checkpoint and
                configuration into a single `.nemo` file. If set to false, uploads the contents of the directory
                containing the model checkpoint and configuration plus additional artifacts.
            model_card (`ModelCard`, *optional*): Model card to upload with the model. If None, will use the model
                card template provided by the class itself.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            The url of the commit of your model in the given repository.
        """
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError("Invalid repo_id provided. Please provide a repo_id of the form `username/repo-name`.")

        domain_name, model_name = repo_id.split("/")

        if token is None:
            token = get_hf_token()

        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            saved_path.mkdir(parents=True, exist_ok=True)

            # Save nemo file in temp dir
            # Get SaveRestoreConnector from subclass implementation
            if not hasattr(self, '_save_restore_connector'):
                raise NotImplementedError(
                    "Model must implement a `_save_restore_connector` property to push to the HuggingFace Hub."
                )

            # We want to save a NeMo file, but not pack its contents into a tarfile by default
            save_restore_connector = self._save_restore_connector
            save_restore_connector.pack_nemo_file = pack_nemo_file

            nemo_filepath = saved_path / f"{model_name}.nemo"
            self.save_to(nemo_filepath)

            # Save model card in temp dir
            if model_card is None:
                card_model_name = model_name.replace("_", " ").split(" ")
                card_model_name = " ".join([word.capitalize() for word in card_model_name])
                template_kwargs = {
                    'model_name': card_model_name,
                    'repo_id': repo_id,
                }
                model_card = self.generate_model_card(template_kwargs=template_kwargs)

            # Convert model card to str
            model_card = str(model_card)

            # Write model card to temp dir
            model_card_filepath = saved_path / f"README.md"
            model_card_filepath.write_text(str(model_card), encoding='utf-8', errors='ignore')

            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    def generate_model_card(
        self, type: str = "hf", template: str = None, template_kwargs: Optional[Dict[str, str]] = None
    ) -> object:
        """
        Generates a ModelCard for the current model. This method is called when pushing the model to the Hub.

        Returns:
            An object that can be represented as a str representation of the model card, usually in Markdown format.
        """
        if template is None:
            template = copy.deepcopy(NEMO_DEFAULT_MODEL_CARD_TEMPLATE)

        # Populate template kwargs with common model card fields
        if template_kwargs is None:
            template_kwargs = {}

        if type == "hf":
            card_data = ModelCardData(
                library_name='nemo', tags=['pytorch', 'NeMo'], license='cc-by-4.0', ignore_metadata_errors=True,
            )

            if 'card_data' not in template_kwargs:
                template_kwargs['card_data'] = card_data.to_yaml()

            # Update template with kwargs
            # We need to do a manual replace because not all keys may be provided in the kwargs
            for key, val in template_kwargs.items():
                template = template.replace("{" + key.strip() + "}", val)

            hf_model_card = ModelCard(template)
            return hf_model_card

        else:
            raise ValueError(f"Model card type {type} not supported.")


class typecheck:
    """
    A decorator which performs input-output neural type checks, and attaches
    neural types to the output of the function that it wraps.

    Requires that the class inherit from :class:`~nemo.core.Typing` in order to perform
    type checking, and will raise an error if that is not the case.

    # Usage (Class level type support)

    .. code-block:: python

        @typecheck()
        def fn(self, arg1, arg2, ...):
            ...

    # Usage (Function level type support)

    .. code-block:: python

        @typecheck(input_types=..., output_types=...)
        def fn(self, arg1, arg2, ...):
            ...

    Points to be noted:

    1) The brackets () in `@typecheck()` are necessary.

        You will encounter a TypeError: __init__() takes 1 positional argument but X
        were given without those brackets.

    2) The function can take any number of positional arguments during definition.

        When you call this function, all arguments must be passed using kwargs only.

    """

    class TypeState(Enum):
        """
        Placeholder to denote the default value of type information provided.
        If the constructor of this decorator is used to override the class level type definition,
        this enum value indicate that types will be overridden.
        """

        UNINITIALIZED = 0

    def __init__(
        self,
        input_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
        output_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
        ignore_collections: bool = False,
    ):
        self.input_types = input_types
        self.output_types = output_types

        if input_types == self.TypeState.UNINITIALIZED:
            self.input_override = False
        else:
            self.input_override = True

        if output_types == self.TypeState.UNINITIALIZED:
            self.output_override = False
        else:
            self.output_override = True

        self.ignore_collections = ignore_collections

    @wrapt.decorator(enabled=is_typecheck_enabled)
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        """
        Wrapper method that can be used on any function of a class that implements :class:`~nemo.core.Typing`.
        By default, it will utilize the `input_types` and `output_types` properties of the class inheriting Typing.

        Local function level overrides can be provided by supplying dictionaries as arguments to the decorator.

        Args:
            input_types: Union[TypeState, Dict[str, NeuralType]]. By default, uses the global `input_types`.
            output_types: Union[TypeState, Dict[str, NeuralType]]. By default, uses the global `output_types`.
            ignore_collections: Bool. Determines if container types should be asserted for depth checks, or
                if depth checks are skipped entirely.

        """
        if instance is None:
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if not isinstance(instance, Typing):
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if hasattr(instance, 'input_ports') or hasattr(instance, 'output_ports'):
            raise RuntimeError(
                "Typing requires override of `input_types()` and `output_types()`, "
                "not `input_ports() and `output_ports()`"
            )

        # Preserve type information
        if self.input_types is typecheck.TypeState.UNINITIALIZED:
            self.input_types = instance.input_types

        if self.output_types is typecheck.TypeState.UNINITIALIZED:
            self.output_types = instance.output_types

        # Resolve global type or local overridden type
        if self.input_override:
            input_types = self.input_types
        else:
            input_types = instance.input_types

        if self.output_override:
            output_types = self.output_types
        else:
            output_types = instance.output_types

        # If types are not defined, skip type checks and just call the wrapped method
        if input_types is None and output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if input_types is not None and len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        instance._validate_input_types(input_types=input_types, ignore_collections=self.ignore_collections, **kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        instance._attach_and_validate_output_types(
            output_types=output_types, ignore_collections=self.ignore_collections, out_objects=outputs
        )

        return outputs

    @staticmethod
    def set_typecheck_enabled(enabled: bool = True):
        """
        Global method to enable/disable typechecking.

        Args:
            enabled: bool, when True will enable typechecking.
        """
        global _TYPECHECK_ENABLED
        _TYPECHECK_ENABLED = enabled

    @staticmethod
    @contextmanager
    def disable_checks():
        """
        Context manager that temporarily disables type checking within its context.
        """
        typecheck.set_typecheck_enabled(enabled=False)
        try:
            yield
        finally:
            typecheck.set_typecheck_enabled(enabled=True)

    @staticmethod
    def set_semantic_check_enabled(enabled: bool = True):
        """
        Global method to enable/disable semantic typechecking.

        Args:
            enabled: bool, when True will enable semantic typechecking.
        """
        global _TYPECHECK_SEMANTIC_CHECK_ENABLED
        _TYPECHECK_SEMANTIC_CHECK_ENABLED = enabled

    @staticmethod
    @contextmanager
    def disable_semantic_checks():
        """
        Context manager that temporarily disables semantic type checking within its context.
        """
        typecheck.set_semantic_check_enabled(enabled=False)
        try:
            yield
        finally:
            typecheck.set_semantic_check_enabled(enabled=True)
