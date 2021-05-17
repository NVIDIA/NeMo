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
from nemo.utils import logging
from nemo.utils.export_utils import replace_for_export

__all__ = ['ExportFormat', 'Exportable']


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {
    ".pt": ExportFormat.TORCHSCRIPT,
    ".ts": ExportFormat.TORCHSCRIPT,
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

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def get_input_names(self):
        if not (hasattr(self, 'input_types')):
            raise NotImplementedError('For export to work you must define input_types')
        input_names = list(self.input_types.keys())
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_input_names:
            input_names.remove(name)
        return input_names

    def get_output_names(self):
        if not (hasattr(self, 'output_types')):
            raise NotImplementedError('For export to work you must define output_types')
        output_names = list(self.output_types.keys())
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_output_names:
            output_names.remove(name)
        return output_names

    def get_input_dynamic_axes(self, input_names):
        dynamic_axes = defaultdict(list)
        for name in input_names:
            dynamic_axes = {
                **dynamic_axes,
                **self._extract_dynamic_axes(name, self.input_types[name]),
            }
        return dynamic_axes

    def get_output_dynamic_axes(self, output_names):
        dynamic_axes = defaultdict(list)
        for name in output_names:
            dynamic_axes = {
                **dynamic_axes,
                **self._extract_dynamic_axes(name, self.output_types[name]),
            }
        return dynamic_axes

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
        dynamic_axes=None,
        check_tolerance=0.01,
        forward_method=None,
    ):
        my_args = locals()
        del my_args['self']

        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        output_descr = qual_name + ' exported to ONNX'

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Allow user to completely override forward method to export
            if forward_method is None and hasattr(type(self), "forward_for_export"):
                forward_method = type(self).forward_for_export

            if forward_method:
                old_forward_method = type(self).forward
                type(self).forward = forward_method

            # Set module to eval mode
            if set_eval:
                self.eval()

            format = self.get_format(output)

            if input_example is None:
                input_example = self.input_module.input_example()

            my_args['input_example'] = input_example

            # Run (posibly overridden) prepare method before calling forward()
            self._prepare_for_export(**my_args)

            input_list = list(input_example)
            input_dict = {}
            # process possible kwargs
            if isinstance(input_list[-1], dict):
                input_dict = input_list[-1]
                input_list = tuple(input_list[:-1])

            input_names = self.input_module.get_input_names()
            output_names = self.output_module.get_output_names()

            output_example = self.forward(*input_list, **input_dict)

            with torch.jit.optimized_execution(True), torch.no_grad():
                jitted_model = None
                if try_script:
                    try:
                        jitted_model = torch.jit.script(self)
                    except Exception as e:
                        print("jit.script() failed!", e)

            with torch.jit.optimized_execution(True), torch.no_grad():
                if format == ExportFormat.TORCHSCRIPT:
                    if jitted_model is None:
                        jitted_model = torch.jit.trace(
                            self,
                            input_example,
                            strict=False,
                            optimize=True,
                            check_trace=check_trace,
                            check_tolerance=check_tolerance,
                        )
                    if verbose:
                        print(jitted_model.code)
                    jitted_model.save(output)
                    assert os.path.exists(output)

                elif format == ExportFormat.ONNX:
                    if jitted_model is None:
                        jitted_model = self

                    # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
                    if dynamic_axes is None and use_dynamic_axes:
                        dynamic_axes = self.input_module.get_input_dynamic_axes(input_names)
                        dynamic_axes = {**dynamic_axes, **self.output_module.get_output_dynamic_axes(output_names)}

                    torch.onnx.export(
                        jitted_model,
                        input_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        export_params=export_params,
                        do_constant_folding=do_constant_folding,
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                        example_outputs=output_example,
                    )

                    # Verify the model can be read, and is valid
                    onnx_model = onnx.load(output)
                    onnx.checker.check_model(onnx_model, full_check=True)

                else:
                    raise ValueError(f'Encountered unknown export format {format}.')
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            if forward_method:
                type(self).forward = old_forward_method
        return ([output], [output_descr])

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

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        replace_1D_2D = kwargs.get('replace_1D_2D', False)
        replace_for_export(self, replace_1D_2D)
