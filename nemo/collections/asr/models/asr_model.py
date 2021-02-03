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
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable

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
        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        output1 = os.path.join(os.path.dirname(output), 'encoder_' + os.path.basename(output))
        output1_descr = qual_name + ' Encoder exported to ONNX'
        if input_example is None:
            input_example = self.encoder.input_example()
        en_out_example = self.encoder.forward(input_example)
        encoder_onnx = self.encoder.export(
            output1,
            input_example=input_example,
            output_example=en_out_example,
            verbose=verbose,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            onnx_opset_version=onnx_opset_version,
            try_script=try_script,
            set_eval=set_eval,
            check_trace=check_trace,
            use_dynamic_axes=use_dynamic_axes,
        )

        output2 = os.path.join(os.path.dirname(output), 'decoder_' + os.path.basename(output))
        output2_descr = qual_name + ' Decoder exported to ONNX'
        decoder_onnx = self.decoder.export(
            output2,
            input_example=en_out_example,
            output_example=output_example,
            verbose=verbose,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            onnx_opset_version=onnx_opset_version,
            try_script=try_script,
            set_eval=set_eval,
            check_trace=check_trace,
            use_dynamic_axes=use_dynamic_axes,
        )

        output_model = attach_onnx_to_onnx(encoder_onnx, decoder_onnx, "DC")
        output_descr = qual_name + ' Encoder+Decoder exported to ONNX'
        onnx.save(output_model, output)
        return ([output, output1, output2], [output_descr, output1_descr, output2_descr])
