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
import os
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Dict

import onnx
import torch

from nemo.core.classes import typecheck
from nemo.core.neural_types import AxisKind, NeuralType
from nemo.utils.export_utils import replace_for_export

__all__ = ['ExportFormat', 'Exportable']


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {
    ".pt": ExportFormat.TORCHSCRIPT,
    ".onnx": ExportFormat.ONNX,
}


class Exportable(ABC):
    """
    This Interface should be implemented by particular classes derived from nemo.core.NeuralModule or nemo.core.ModelPT.
    It gives these entities ability to be exported for deployment to formats such as ONNX.
    """

    @staticmethod
    def get_format(filename: str):
        _, ext = os.path.splitext(filename)
        try:
            return _EXT_DICT[ext]
        except KeyError:
            raise ValueError(f"Export file {filename} extension does not correspond to any export format!")

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 12,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = True,
        use_dynamic_axes: bool = True,
    ):
        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Set module to eval mode
            if set_eval:
                self.eval()

            format = self.get_format(output)
            self._prepare_for_export()

            if input_example is not None:
                _in_example = input_example
            else:
                _in_example = self.input_example()

            if output_example is None:
                _out_example = self.forward(*_in_example)

            if not (hasattr(self, 'input_types') and hasattr(self, 'output_types')):
                raise NotImplementedError('For export to work you must define input and output types')
            input_names = list(self.input_types.keys())
            output_names = list(self.output_types.keys())
            # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
            dynamic_axes = defaultdict(list)

            # extract dynamic axes and remove unnecessary inputs/outputs
            # for input_ports
            for _name, ntype in self.input_types.items():
                if _name in self.disabled_deployment_input_names:
                    input_names.remove(_name)
                    continue
                if use_dynamic_axes:
                    dynamic_axes = {**dynamic_axes, **self._extract_dynamic_axes(_name, ntype)}
            # for output_ports
            for _name, ntype in self.output_types.items():
                if _name in self.disabled_deployment_output_names:
                    output_names.remove(_name)
                    continue
                if use_dynamic_axes:
                    dynamic_axes = {**dynamic_axes, **self._extract_dynamic_axes(_name, ntype)}

            if len(dynamic_axes) == 0:
                dynamic_axes = None

            with torch.jit.optimized_execution(True):
                jitted_model = None
                if try_script:
                    try:
                        jitted_model = torch.jit.script(self)
                    except Exception as e:
                        print("jit.script() failed!", e)
                if _in_example is None:
                    raise ValueError(f'Example input is None, but jit.script() has failed or not tried')

                if isinstance(_in_example, Dict):
                    _in_example = tuple(_in_example.values())

                if jitted_model is None:
                    jitted_model = torch.jit.trace(self, _in_example, check_trace=check_trace)

                if format == ExportFormat.TORCHSCRIPT:
                    jitted_model.save(output)
                    assert os.path.exists(output)
                elif format == ExportFormat.ONNX:
                    if _out_example is None:
                        if isinstance(_in_example, tuple):
                            _out_example = self.forward(*_in_example)
                        else:
                            _out_example = self.forward(_in_example)

                    torch.onnx.export(
                        jitted_model,
                        _in_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        export_params=export_params,
                        do_constant_folding=do_constant_folding,
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                        example_outputs=_out_example,
                    )

                    # Verify the model can be read, and is valid
                    onnx_model = onnx.load(output)
                    onnx.checker.check_model(onnx_model, full_check=True)
                    return onnx_model
                else:
                    raise ValueError(f'Encountered unknown export format {format}.')
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
        return [output]  # Subclasses may create more than one file.

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set()

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set()

    @property
    def supported_export_formats(self):
        """Implement this method to return a set of export formats supported. Default is all types."""
        return set([ExportFormat.ONNX, ExportFormat.TORCHSCRIPT])

    @staticmethod
    def _extract_dynamic_axes(name: str, ntype: NeuralType):
        """
        Implement this method to provide dynamic axes id for ONNX export.
        By default, this method will extract BATCH and TIME dimension ids from each provided input/output name argument.

        For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
        shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
        as they can change from call to call during inference.

        Args:
            name: Name of input or output parameter
            ntype: Corresponding Neural Type

        Returns:

        """
        dynamic_axes = defaultdict(list)
        if ntype.axes:
            for ind, axis in enumerate(ntype.axes):
                if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                    dynamic_axes[name].append(ind)
        return dynamic_axes

    def _prepare_for_export(self):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        replace_for_export(self)
