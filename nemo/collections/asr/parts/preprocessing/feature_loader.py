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
import numpy as np
import torch


class ExternalFeatureLoader(object):
    """Feature loader that load external features store in certain format. 
    Currently support pickle, npy and npz format.
    """

    def __init__(self, file_path, sample_rate=16000, augmentor=None):
        """Load feature samples from file_path.
        file_path (str) is the path of the file that stores feature/sample.
        """
        self._file_path = file_path
        self.augmentor = augmentor

    def load_feature_from_file(self, file_path):
        """Load samples from file_path and convert it to be of type float32
        file_path (str) is the path of the file that stores feature/sample.
        """
        # load pickle/npy/npz file
        try:
            samples = np.load(file_path, allow_pickle=True)
            return self._convert_samples_to_float32(samples)
        except:
            raise TypeError(
                "Error open feature files. Probably not supported the format yet. Support pkl, npz, and npy format now. "
            )
        # TODO load other type of files such as kaldi io ark

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    def process(self, file_path, offset=0, duration=0, trim=False, orig_sr=None):
        feature = self.load_feature_from_file(file_path)
        return self.process_segment(feature)

    def process_segment(self, feature_segment):
        if self.augmentor:
            # augmentor for external features. Here possible augmentor for external embedding feature is Diaconis Augmentation and might be implemented later
            self.augmentor.perturb(feature_segment)
            return torch.tensor(feature_segment, dtype=torch.float)

        return torch.tensor(feature_segment, dtype=torch.float)
