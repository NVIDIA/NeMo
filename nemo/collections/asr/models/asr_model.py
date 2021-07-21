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
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from omegaconf import OmegaConf

from nemo.core.classes import ModelPT, exportable, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.utils import model_utils

__all__ = ['ASRModel']


class ASRModel(ModelPT, ABC):
    @abstractmethod
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        """
        Takes paths to audio files and returns text transcription
        Args:
            paths2audio_files: paths to audio fragment to be transcribed

        Returns:
            transcription texts
        """
        pass

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_wer': wer_num / wer_denom}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, 'test_wer': wer_num / wer_denom}
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # recursively walk the subclasses to generate pretrained model info
        list_of_models = model_utils.resolve_subclass_pretrained_model_info(cls)
        return list_of_models


class ExportableEncDecModel(Exportable):
    """
    Simple utiliy mix-in to export models that consist of encoder/decoder pair 
    plus pre/post processor, but have to be exported as encoder/decoder pair only
    (covers most ASR classes)
    """

    @property
    def input_module(self):
        return self.encoder

    @property
    def output_module(self):
        return self.decoder

    def forward_for_export(self, input, length=None):
        encoder_output = self.input_module(input, length)
        if isinstance(encoder_output, tuple):
            return self.output_module(encoder_output[0])
        else:
            return self.output_module(encoder_output)

    def _prepare_for_export(self, **kwargs):
        self.input_module._prepare_for_export(**kwargs)
        self.output_module._prepare_for_export(**kwargs)


class ExportableEncDecJointModel(Exportable):
    """
    Simple utiliy mix-in to export models that consist of encoder/decoder pair
    plus pre/post processor, but have to be exported as encoder/decoder pair only
    (covers most ASR classes)
    """

    @property
    def input_module(self):
        return self.encoder

    @property
    def output_module(self):
        return self.decoder

    @property
    def joint_module(self):
        return self.joint

    @property
    def disabled_deployment_input_names(self):
        encoder_names = self.input_module.disabled_deployment_input_names
        decoder_names = self.output_module.disabled_deployment_input_names
        joint_names = self.joint_module.disabled_deployment_input_names
        return set.union(encoder_names, decoder_names, joint_names)

    @property
    def disabled_deployment_output_names(self):
        encoder_names = self.input_module.disabled_deployment_output_names
        decoder_names = self.output_module.disabled_deployment_output_names
        joint_names = self.joint_module.disabled_deployment_output_names
        return set.union(encoder_names, decoder_names, joint_names)

    def forward_for_encoder_export(self, encoder_inputs, length):
        encoder_output = self.input_module(encoder_inputs, length)
        encoder_output, encoded_length = encoder_output

        return encoder_output, encoded_length

    def forward_for_decoder_joint_export(self, encoder_output, decoder_inputs, decoder_lengths, state_h, state_c):
        decoder, joint = self.output_module, self.joint_module

        if state_h is not None and state_c is not None:
            states = (state_h, state_c)
        else:
            states = None

        decoder_outputs = decoder(decoder_inputs, decoder_lengths, states)
        decoder_output = decoder_outputs[0]
        decoder_length = decoder_outputs[1]
        state_h, state_c = decoder_outputs[2][0], decoder_outputs[2][1]

        joint_output = joint(encoder_output, decoder_output)
        return (joint_output, decoder_length, state_h, state_c)

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 13,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):
        my_args = locals()
        del my_args['self']

        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        output_descr = qual_name + ' exported to ONNX'

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Set module to eval mode
            self._set_eval(set_eval)

            format = self.get_format(output)

            # Assign special flag for RNNT export of encoder
            if not hasattr(self.input_module, '_rnnt_export'):
                raise ValueError(
                    f"{self.input_module.__class__.__name__} must have a bool attribute `_rnnt_export`, "
                    f"which is necessary for RNNT export."
                )

            if not hasattr(self.output_module, '_rnnt_export'):
                raise ValueError(
                    f"{self.output_module.__class__.__name__} must have a bool attribute `_rnnt_export`, "
                    f"which is necessary for RNNT export."
                )

            if not hasattr(self.joint_module, '_rnnt_export'):
                raise ValueError(
                    f"{self.joint_module.__class__.__name__} must have a bool attribute `_rnnt_export`, "
                    f"which is necessary for RNNT export."
                )

            self.input_module._rnnt_export = True
            self.output_module._rnnt_export = True
            self.joint_module._rnnt_export = True

            if input_example is None:
                encoder_examples, decoder_examples = self._get_input_example()
                input_example = [encoder_examples, decoder_examples]
            else:
                assert type(input_example) in (list, tuple) and len(input_example) == 2, (
                    "input_example must " "be a list of two tensors," "for encoder and decoder input"
                )
                encoder_examples, decoder_examples = input_example

            if output_example is not None:
                assert type(output_example) in (list, tuple) and len(output_example) == 2, (
                    "output_example must " "be a list of two tensors," "for encoder and decoder+joint" " output"
                )
                encoder_output_example, decoder_joint_output_example = output_example
            else:
                encoder_output_example = None
                decoder_joint_output_example = None

            my_args['input_example'] = input_example

            # Run (posibly overridden) prepare method before calling forward()
            self._prepare_for_export(**my_args)

            encoder_input_list, encoder_input_dict = self._setup_input_example(encoder_examples)
            decoder_input_list, decoder_input_dict = self._setup_input_example(decoder_examples)

            encoder_input_names, decoder_input_names = self._process_input_names()
            encoder_output_names, decoder_output_names, joint_output_names = self._process_output_names()

            # process decoder states; by convension states must be the last in the list and must be wrapped in a tuple
            (
                decoder_input_list,
                decoder_input_names,
                input_state_names,
                num_states,
                output_state_names,
                state_names,
            ) = self._process_states_names(decoder_input_list, decoder_input_names)

            with torch.jit.optimized_execution(True), torch.no_grad():
                # Encoder export
                encoder_jitted_model = self._try_jit_compile_model(self.input_module, try_script)

                if format == exportable.ExportFormat.TORCHSCRIPT:
                    raise NotImplementedError()
                    # # Allow user to completely override forward method to export
                    # forward_method, original_forward_method = self._wrap_forward_method('encoder')
                    # encoder_output_example = self.forward(*encoder_input_list, **encoder_input_dict)
                    #
                    # self._export_torchscript(
                    #     encoder_jitted_model,
                    #     self._augment_output_filename(output, "Encoder"),
                    #     encoder_input_dict,
                    #     encoder_input_list,
                    #     check_trace,
                    #     check_tolerance,
                    #     verbose,
                    # )
                    #
                    # self._export_flag_module = 'decoder_joint'
                    #
                    # # Extract just the encoder logits and remove the encoder lengths
                    # if type(encoder_output_example) in (list, tuple):
                    #     encoder_output_example = encoder_output_example[0]
                    #
                    # encoder_decoder_input_list = [encoder_output_example] + list(decoder_input_list)
                    # encoder_decoder_input_dict = decoder_input_dict
                    #
                    # encoder_decoder_input_list = tuple(encoder_decoder_input_list)
                    #
                    # # Allow user to completely override forward method to export
                    # forward_method, _ = self._wrap_forward_method('decoder_joint')
                    # decoder_joint_output_example = self.forward(*encoder_decoder_input_list, **encoder_decoder_input_dict)
                    # decoder_joint_output_example = tuple(decoder_joint_output_example)
                    #
                    # # Resolve output states
                    # if num_states > 0:
                    #     if type(decoder_joint_output_example[-1]) == tuple:
                    #         raise TypeError("Since input states are available, forward must emit flattened states")
                    #
                    #     # remove the name of the states
                    #     logging.info(
                    #         f"Replacing output state name {decoder_output_names[-1]} with {str(output_state_names)}"
                    #     )
                    #     decoder_output_names = decoder_output_names[:-1]
                    #
                    # self._export_torchscript(
                    #     None,
                    #     self._augment_output_filename(output, "Decoder-Joint"),
                    #     encoder_decoder_input_dict,
                    #     encoder_decoder_input_list,
                    #     check_trace,
                    #     check_tolerance,
                    #     verbose,
                    # )

                elif format == exportable.ExportFormat.ONNX:
                    # Allow user to completely override forward method to export
                    forward_method, original_forward_method = self._wrap_forward_method('encoder')
                    encoder_output_example = self.forward(*encoder_input_list, **encoder_input_dict)

                    self._export_flag_module = 'encoder'

                    self._export_onnx(
                        encoder_jitted_model,
                        encoder_examples,
                        encoder_output_example,
                        encoder_input_names,
                        encoder_output_names,
                        use_dynamic_axes,
                        do_constant_folding,
                        dynamic_axes,
                        self._augment_output_filename(output, "Encoder"),
                        export_params,
                        keep_initializers_as_inputs,
                        onnx_opset_version,
                        verbose,
                    )

                    # Verify the model can be read, and is valid
                    self._verify_onnx_export(
                        self._augment_output_filename(output, "Encoder"),
                        encoder_output_example,
                        encoder_input_list,
                        encoder_input_dict,
                        encoder_input_names,
                        check_tolerance,
                        check_trace,
                    )

                    self._export_flag_module = 'decoder_joint'

                    # Extract just the encoder logits and remove the encoder lengths
                    if type(encoder_output_example) in (list, tuple):
                        encoder_output_example = encoder_output_example[0]

                    encoder_decoder_input_list = [encoder_output_example] + list(decoder_input_list)
                    encoder_decoder_input_dict = decoder_input_dict

                    encoder_decoder_input_list = tuple(encoder_decoder_input_list)

                    # Allow user to completely override forward method to export
                    forward_method, _ = self._wrap_forward_method('decoder_joint')
                    decoder_joint_output_example = self.forward(
                        *encoder_decoder_input_list, **encoder_decoder_input_dict
                    )
                    decoder_joint_output_example = tuple(decoder_joint_output_example)

                    # Resolve output states
                    if num_states > 0:
                        if type(decoder_joint_output_example[-1]) == tuple:
                            raise TypeError("Since input states are available, forward must emit flattened states")

                        # remove the name of the states
                        logging.info(
                            f"Replacing output state name {decoder_output_names[-1]} with {str(output_state_names)}"
                        )
                        decoder_output_names = decoder_output_names[:-1]

                    self._export_onnx(
                        None,
                        encoder_decoder_input_list,
                        decoder_joint_output_example,
                        self._join_input_output_names(["encoder_outputs"], decoder_input_names, input_state_names),
                        self._join_input_output_names(joint_output_names, decoder_output_names, output_state_names),
                        use_dynamic_axes,
                        do_constant_folding,
                        dynamic_axes,
                        self._augment_output_filename(output, "Decoder-Joint"),
                        export_params,
                        keep_initializers_as_inputs,
                        onnx_opset_version,
                        verbose,
                    )

                    # Verify the model can be read, and is valid
                    self._verify_onnx_export(
                        self._augment_output_filename(output, "Decoder-Joint"),
                        decoder_joint_output_example,
                        encoder_decoder_input_list,
                        encoder_decoder_input_dict,
                        self._join_input_output_names(["encoder_outputs"], decoder_input_names, input_state_names),
                        check_tolerance,
                        check_trace,
                    )

                else:
                    raise ValueError(f'Encountered unknown export format {format}.')

        except Exception as e:
            raise e
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            logging.warning(
                "PyTorch Model has been significantly modified. In order to utilize model, delete this "
                "instance and create a new model."
            )

            # replace forward method with original forward method

            type(self).forward = original_forward_method

            # Reset special flag for RNNT export of encoder
            self.input_module._rnnt_export = False
            self.output_module._rnnt_export = False
            self.joint_module._rnnt_export = False

        return ([output], [output_descr])

    def _process_states_names(self, decoder_input_list, decoder_input_names):
        if type(decoder_input_list[-1]) in (list, tuple):
            num_states = len(decoder_input_list[-1])
            states = decoder_input_list[-1]
            decoder_input_list = decoder_input_list[:-1]

            # unpack states
            for state in states:
                decoder_input_list.append(state)

            state_name = decoder_input_names[-1]
            decoder_input_names = decoder_input_names[:-1]
            state_names = [f"{state_name}-{idx + 1}" for idx in range(num_states)]
            input_state_names = ["input-" + name for name in state_names]
            output_state_names = ["output-" + name for name in state_names]

        else:
            num_states = 0
            state_name, state_names, input_state_names, output_state_names = None, [], [], []
        return decoder_input_list, decoder_input_names, input_state_names, num_states, output_state_names, state_names

    def _export_onnx(
        self,
        jitted_model,
        input_example,
        output_example,
        input_names,
        output_names,
        use_dynamic_axes,
        do_constant_folding,
        dynamic_axes,
        output,
        export_params,
        keep_initializers_as_inputs,
        onnx_opset_version,
        verbose,
    ):
        if jitted_model is None:
            jitted_model = self

        dynamic_axes = self._get_dynamic_axes(dynamic_axes, input_names, output_names, use_dynamic_axes)

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

    def _prepare_for_export(self, **kwargs):
        self.input_module._prepare_for_export(**kwargs)
        self.output_module._prepare_for_export(**kwargs)
        self.joint_module._prepare_for_export(**kwargs)

    def _get_input_example(self):
        encoder_examples = super()._get_input_example()
        decoder_examples = self.output_module.input_example()
        return encoder_examples, decoder_examples

    def _set_eval(self, set_eval):
        super()._set_eval(set_eval)
        if set_eval:
            self.joint_module.freeze()

    def _wrap_forward_method(self, method_type: str):
        assert method_type in ['encoder', 'decoder_joint']
        old_forward_method = None

        if hasattr(type(self), "forward_for_encoder_export") and method_type == "encoder":
            encoder_forward_method = type(self).forward_for_encoder_export
            forward_method = encoder_forward_method
            old_forward_method = type(self).forward
            type(self).forward = encoder_forward_method

        elif hasattr(type(self), "forward_for_decoder_joint_export") and method_type == "decoder_joint":
            decoder_joint_forward_method = type(self).forward_for_decoder_joint_export
            forward_method = decoder_joint_forward_method
            old_forward_method = type(self).forward
            type(self).forward = decoder_joint_forward_method

        else:
            forward_method = None

        return forward_method, old_forward_method

    def _process_input_names(self):
        encoder_input_names = super()._process_input_names()
        decoder_input_names = exportable.get_input_names(self.output_module)

        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_input_names:
            if name in encoder_input_names:
                encoder_input_names.remove(name)

            if name in decoder_input_names:
                decoder_input_names.remove(name)
        return encoder_input_names, decoder_input_names

    def _process_output_names(self):
        encoder_output_names = exportable.get_output_names(self.input_module)
        decoder_output_names = exportable.get_output_names(self.output_module)
        joint_output_names = exportable.get_output_names(self.joint_module)

        for name in self.disabled_deployment_output_names:
            if name in encoder_output_names:
                encoder_output_names.remove(name)

            if name in decoder_output_names:
                decoder_output_names.remove(name)

            if name in joint_output_names:
                joint_output_names.remove(name)

        return encoder_output_names, decoder_output_names, joint_output_names

    def _get_dynamic_axes(self, dynamic_axes, input_names, output_names, use_dynamic_axes):
        # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
        if dynamic_axes is None and use_dynamic_axes:
            dynamic_axes = super()._get_dynamic_axes(dynamic_axes, input_names, output_names, use_dynamic_axes)

            if self._export_flag_module == 'encoder':
                return dynamic_axes
            else:
                dynamic_axes = {
                    **dynamic_axes,
                    **exportable.get_input_dynamic_axes(self.output_module, input_names),
                }
                dynamic_axes = {
                    **dynamic_axes,
                    **exportable.get_input_dynamic_axes(self.joint_module, input_names),
                }
                dynamic_axes = {
                    **dynamic_axes,
                    **exportable.get_output_dynamic_axes(self.joint_module, output_names),
                }
                dynamic_axes = self._get_state_dynamic_axes(dynamic_axes, input_names, output_names)

        return dynamic_axes

    def _get_state_dynamic_axes(self, dynamic_axes, input_names, output_names):
        reduced_input_names = []
        for name in input_names:
            if 'input-' in name:
                name = name.replace('input-', '')
                name = re.sub(r"-[0-9]*", "", name)
                reduced_input_names.append(name)

        state_dynamic_axes = {
            **exportable.get_input_dynamic_axes(self.output_module, reduced_input_names),
        }
        input_state_dynamic_axes = {}

        for name in input_names:
            for reduced_name in reduced_input_names:
                if reduced_name in name:
                    input_state_dynamic_axes[name] = state_dynamic_axes[reduced_name]

        dynamic_axes = {**dynamic_axes, **input_state_dynamic_axes}

        # Process output states dynamic axes

        reduced_output_names = []
        for name in output_names:
            if 'output-' in name:
                name = name.replace('output-', '')
                name = re.sub(r"-[0-9]*", "", name)
                reduced_output_names.append(name)

        state_dynamic_axes = {
            **exportable.get_output_dynamic_axes(self.output_module, reduced_output_names),
        }
        output_state_dynamic_axes = {}

        for name in output_names:
            for reduced_name in reduced_output_names:
                if reduced_name in name:
                    output_state_dynamic_axes[name] = state_dynamic_axes[reduced_name]

        dynamic_axes = {**dynamic_axes, **output_state_dynamic_axes}

        return dynamic_axes

    def _augment_output_filename(self, output, prepend: str):
        path, filename = os.path.split(output)
        filename = f"{prepend}-{filename}"
        return os.path.join(path, filename)

    def _join_input_output_names(self, *lists):
        data = []
        for list_ in lists:
            for name in list_:
                if name not in data:
                    data.append(name)
        return data
