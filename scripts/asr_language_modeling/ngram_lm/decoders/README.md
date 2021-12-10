This beam search decoder implementation was originally taken from Baidu DeepSpech project:
https://github.com/PaddlePaddle/DeepSpeech/

Git commit: https://github.com/PaddlePaddle/DeepSpeech/commit/a76fc692123e97e5f183c3a00e30b7f2e2d3f07c


Copyright 2019 Baidu, Inc.
Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Changes:
	modified:   decoder_utils.cpp
        modified:   setup.py
        modified:   setup.sh
	modified:   ctc_beam_search_decoder.cpp
        modified:   ctc_beam_search_decoder.h
         renamed:   swig_decoders.py -> ctc_decoders.py
        modified:   ctc_decoders.py

