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

# WIP. Ignore. The plan is to have non-PTL dependent inference

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.metrics.wer import WER, word_error_rate


def main():
    asr_model = EncDecCTCModel.restore_from(restore_path="/Users/okuchaiev/Workspace/NeMo1.0.0.beta/CheckPoints/WeightConversion/QN15x5En/QuartzNet15x5Base-En.nemo")
    asr_model.eval()
    asr_model.setup_test_data(test_data_config={
        'manifest_filepath': '/Users/okuchaiev/Data/an4_dataset/an4_val.json',
        'sample_rate': 16000,
        'labels': asr_model.decoder.vocabulary,
        'batch_size': 1
    })
    wer = WER(vocabulary=asr_model.decoder.vocabulary)
    test_outs = []
    for test_batch in asr_model.test_dataloader():
        log_probs, encoded_len, greedy_predictions = asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
        test_outs.append(wer.ctc_decoder_predictions_tensor(greedy_predictions))
    print(test_outs)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
