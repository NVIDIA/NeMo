# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from nemo_text_processing.text_normalization.normalize import Normalizer


class ElectronicNormalizer(Normalizer):
    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
    ):

        super().__init__(
            input_case=input_case,
            lang=lang,
            deterministic=deterministic,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
        )
        from nn_wfst.en.electronic.tokenize_and_classify import ClassifyFst
        from nn_wfst.en.electronic.verbalize_final import VerbalizeFinalFst

        self.tagger = self.tagger = ClassifyFst(
            input_case=input_case, deterministic=deterministic, cache_dir=cache_dir, overwrite_cache=overwrite_cache
        )
        self.verbalizer = VerbalizeFinalFst(deterministic=deterministic)


if __name__ == "__main__":
    import sys

    input_string = sys.argv[1]
    normalizer = ElectronicNormalizer(input_case='cased')
    print(normalizer.normalize(input_string, verbose=False))
