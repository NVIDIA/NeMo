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

import os
import pytest
import unittest

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
from nemo.collections.tts.g2p.modules import IPAG2P
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.tts.inference.text_processors import BaseTextProcessor, IPATextTokenizer

from nemo.collections.tts.inference.pipeline import TTSPipeline
from nemo.collections.tts.inference.spectrogram_synthesizers import FastPitchSpectrogramSynthesizer
from nemo.collections.tts.inference.vocoders import HifiGanVocoder
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel


class TestTTSPipeline(unittest.TestCase):

    PHONEME_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict", "test_dict.txt")

    @classmethod
    def setUpClass(cls):
        super(TestTTSPipeline, cls).setUpClass()

        normalizer = Normalizer(lang="en", input_case="cased")
        cls.text_processor = BaseTextProcessor(normalizer)

        g2p = IPAG2P(TestTTSPipeline.PHONEME_DICT_PATH, locale="en-US", apply_to_oov_word=lambda x: x, use_chars=True,)
        ipa_tokenizer = IPATokenizer(g2p=g2p)
        cls.ipa_text_tokenizer = IPATextTokenizer(tokenizer=ipa_tokenizer)
        cls.g2p = cls.ipa_text_tokenizer

        fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker").eval().to("cpu")
        cls.spec_synthesizer = FastPitchSpectrogramSynthesizer(model=fastpitch_model)

        hifigan_model = HifiGanModel.from_pretrained("tts_en_hifitts_hifigan_ft_fastpitch").eval().to("cpu")
        cls.vocoder = HifiGanVocoder(model=hifigan_model)

        cls.tts_pipeline = TTSPipeline(
            text_processor=cls.text_processor,
            g2p=cls.ipa_text_tokenizer,
            text_tokenizer=cls.ipa_text_tokenizer,
            spectrogram_synthesizer=cls.spec_synthesizer,
            vocoder=cls.vocoder,
        )

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_pipeline(self):
        input_text = "Hello  world 0!"
        pipeline = TTSPipeline()

        with pytest.raises(AssertionError):
            pipeline.text_to_speech(text=input_text)

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_process_text(self):
        input_text = "Hello  world 0!"
        expected_output = "Hello world zero!"

        processed_text = self.tts_pipeline.process_text(text=input_text)

        assert processed_text == expected_output

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_process_text_only_text_modules(self):
        input_text = "Hello  world 1!"
        expected_output = "Hello world one!"
        pipeline = TTSPipeline(
            text_processor=self.text_processor, g2p=self.g2p, text_tokenizer=self.ipa_text_tokenizer
        )

        processed_text = pipeline.process_text(text=input_text)

        assert processed_text == expected_output

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_text_to_speech(self):
        input_text = "Hello world"
        speaker = 1
        pitch = 0.1

        audio = self.tts_pipeline.text_to_speech(text=input_text, speaker=speaker, pitch=pitch)

        assert len(audio.shape) == 1
        assert audio.shape[0] > len(input_text)
