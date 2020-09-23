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

"""
WARNING: Running this test will download ALL pre-trained NeMo models.
This is bandwidth and disk space consuming.
"""

import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts

for refresh_cache in [False]:
    # Test ASR collection
    print(nemo_asr.models.EncDecCTCModel.list_available_models())
    for model_name in [
        'QuartzNet15x5Base-En',
        'QuartzNet15x5Base-Zh',
        'QuartzNet5x5LS-En',
        'QuartzNet15x5NR-En',
        'Jasper10x5Dr-En',
    ]:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_asr.models.EncDecCTCModelBPE.list_available_models())
    for model_name in ['ContextNet-192-WPE-1024-8x-Stride']:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_asr.models.EncDecClassificationModel.list_available_models())
    for model_name in [
        'MatchboxNet-3x1x64-v1',
        'MatchboxNet-3x2x64-v1',
        'MatchboxNet-3x1x64-v2',
        'MatchboxNet-3x1x64-v2',
        'MatchboxNet-3x1x64-v2-subset-task',
        'MatchboxNet-3x2x64-v2-subset-task',
        'MatchboxNet-VAD-3x2',
    ]:
        model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            model_name=model_name, refresh_cache=refresh_cache
        )
    print(nemo_asr.models.EncDecSpeakerLabelModel.list_available_models())
    for model_name in ['SpeakerNet_recognition', 'SpeakerNet_verification']:
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=model_name, refresh_cache=refresh_cache
        )

    # Test NLP collection
    print(nemo_nlp.models.TokenClassificationModel.list_available_models())
    for model_name in ['NERModel']:
        model = nemo_nlp.models.TokenClassificationModel.from_pretrained(
            model_name=model_name, refresh_cache=refresh_cache
        )
    print(nemo_nlp.models.PunctuationCapitalizationModel.list_available_models())
    for model_name in ['Punctuation_Capitalization_with_BERT', 'Punctuation_Capitalization_with_DistilBERT']:
        model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
            model_name=model_name, refresh_cache=refresh_cache
        )
    print(nemo_nlp.models.QAModel.list_available_models())
    for model_name in [
        'BERTBaseUncasedSQuADv1.1',
        'BERTBaseUncasedSQuADv2.0',
        'BERTLargeUncasedSQuADv1.1',
        'BERTLargeUncasedSQuADv2.0',
    ]:
        model = nemo_nlp.models.QAModel.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_nlp.models.IntentSlotClassificationModel.list_available_models())
    for model_name in ['Joint_Intent_Slot_Assistant']:
        model = nemo_nlp.models.IntentSlotClassificationModel.from_pretrained(
            model_name=model_name, refresh_cache=refresh_cache
        )

    # Test TTS collection
    print(nemo_tts.models.Tacotron2Model.list_available_models())
    for model_name in ['Tacotron2-22050Hz']:
        model = nemo_tts.models.Tacotron2Model.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_tts.models.WaveGlowModel.list_available_models())
    for model_name in ['WaveGlow-22050Hz']:
        model = nemo_tts.models.WaveGlowModel.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_tts.models.SqueezeWaveModel.list_available_models())
    for model_name in ['SqueezeWave-22050Hz']:
        model = nemo_tts.models.SqueezeWaveModel.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)
    print(nemo_tts.models.GlowTTSModel.list_available_models())
    for model_name in ['GlowTTS-22050Hz']:
        model = nemo_tts.models.GlowTTSModel.from_pretrained(model_name=model_name, refresh_cache=refresh_cache)

print("############ THAT'S ALL FOLKS! ############")
