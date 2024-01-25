# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch.utils.data
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper


class CanaryDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for Canary model, which has an encoder-decoder architecture.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer._tokenizer.pad_id

    def __getitem__(self, cuts) -> tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)

        tokens = [self.tokenizer(c.supervisions[0].text, c.supervisions[0].language) for c in cuts]
        tokens = self._canary_format(tokens, cuts)
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)

        return audio, audio_lens, tokens, token_lens

    def _canary_format(self, tokens, cuts):
        """
        prepend and append control tokens to the token sequence as per canary format

        Format:
        sot, src_lang_id/no_speech, transcribe/translate, tgt_lang_id, text, eot
        """
        canary_tokens = []
        for t, c in zip(tokens, cuts):
            if isinstance(c, MixedCut):
                c = c._first_non_padding_cut
            assert isinstance(c, MonoCut), "Expected MonoCut."

            c_t = []  # canary_tokens for this cut

            # bos
            c_t.append(self.tokenizer._tokenizer.bos_id)

            # if len(t) is 0 append no-speech token
            if len(t) == 0:
                c_t.append(self.tokenizer._tokenizer.nospeech_id)
            else:
                # src_lang_id/no_speech
                src_lang_id = self.tokenizer._tokenizer.to_language_id(c.custom['source_lang'])
                c_t.append(src_lang_id)

                # task
                task = c.custom['taskname']
                if task == 'asr':
                    c_t.append(self.tokenizer._tokenizer.transcribe_id)
                elif task == 's2t_translation':
                    c_t.append(self.tokenizer._tokenizer.translate_id)
                else:
                    raise ValueError(f"Unknown task: {task}")

                # tgt_lang_id
                tgt_lang_id = self.tokenizer._tokenizer.to_language_id(c.custom['target_lang'])
                c_t.append(tgt_lang_id)

                # PnC
                pnc = f"{c.custom['pnc']}".lower().strip()  # to account for bool or str
                if pnc in set(['yes', 'true']):
                    c_t.append(self.tokenizer._tokenizer.pnc_id)
                elif pnc in set(['no', 'false']):
                    c_t.append(self.tokenizer._tokenizer.nopnc_id)
                else:
                    raise ValueError(f"Unknown PnC: {pnc}")

                # text
                c_t.extend(t)

            # eos
            c_t.append(self.tokenizer._tokenizer.eos_id)

            canary_tokens.append(c_t)

        return canary_tokens
