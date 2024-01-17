from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse import cut

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self, tokenizer, noise_cuts: Optional = None, force_strip_pnc: bool = False, token_sequence_format: str = None,
    ):
        from lhotse.dataset import AudioSamples, CutMix

        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.maybe_mix_noise = (
            _identity if noise_cuts is None else CutMix(noise_cuts, pad_to_longest=False, random_mix_offset=True)
        )
        self.force_strip_pnc = force_strip_pnc

        if token_sequence_format is not None:
            assert token_sequence_format in ['canary'], f"Unsupported token_sequence_format: {token_sequence_format}"
        self.token_sequence_format = token_sequence_format

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        from lhotse.dataset.collation import collate_vectors

        # commenting this, as we need to maintain the order of cuts in the batch during infernce.
        # and sorting is not needed for training.
        # cuts = cuts.sort_by_duration()
        cuts = self.maybe_mix_noise(cuts)
        audio, audio_lens, cuts = self.load_audio(cuts)

        if self.force_strip_pnc:
            # Note(pzelasko): this is canary-specific temporary hack to check that PNC does not break things.
            for c in cuts:
                c.supervisions[0].text = (
                    c.supervisions[0].text.replace(".", "").replace(",", "").replace("?", "").lower()
                )

        tokens = [
            self.tokenizer(
                c.supervisions[0].text, 'en' if c.supervisions[0].language is None else c.supervisions[0].language
            )
            for c in cuts
        ]
        if self.token_sequence_format == 'canary':
            tokens = self._canary_format(tokens, cuts)
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)

        if self.token_sequence_format == 'canary':
            padding_value = self.tokenizer._tokenizer.pad_id
        else:
            padding_value = 0
        tokens = collate_vectors(tokens, padding_value=padding_value)

        return audio, audio_lens, tokens, token_lens

    def _canary_format(self, tokens, cuts):
        """
        prepend and append control tokens to the token sequence as per canary format
        
        Format:
        sot, src_lang_id/no_speech, transcribe/translate, tgt_lang_id, text, eot
        """
        canary_tokens = []
        for t, c in zip(tokens, cuts):
            if isinstance(c, cut.MixedCut):
                c = c._first_non_padding_cut
            assert isinstance(c, cut.MonoCut), "Expected MonoCut."

            c_t = []  # canary_tokens for this cut

            # bos
            c_t.append(self.tokenizer._tokenizer.bos_id)

            # if len(t) is 0 append no-speech token
            if len(t) == 0:
                c_t.append(self.tokenizer._tokenizer.nospeech_id)
            else:
                # src_lang_id/no_speech
                src_lang_id = self.tokenizer._tokenizer.to_language_id(c.custom.get('source_lang', 'en'))
                c_t.append(src_lang_id)

                # task
                task = c.custom.get('taskname', 'asr')
                if task == 'asr':
                    c_t.append(self.tokenizer._tokenizer.transcribe_id)
                elif task == 's2t_translation':
                    c_t.append(self.tokenizer._tokenizer.translate_id)
                else:
                    raise ValueError(f"Unknown task: {task}")

                # tgt_lang_id
                tgt_lang_id = self.tokenizer._tokenizer.to_language_id(c.custom.get('target_lang', 'en'))
                c_t.append(tgt_lang_id)

                # PnC
                pnc = f"{c.custom.get('pnc', 'no')}".lower().strip()  # to account for bool or str
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


class TokenizerWrapper:
    """
    Provide a unified interface for NeMo Tokenizer, AggregateTokenizer, and (char) Parser.
    """

    def __init__(self, tokenizer):
        from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
        from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

        self._tokenizer = tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            self._impl = self._call_agg_tokenizer
        elif isinstance(tokenizer, TokenizerSpec):
            self._impl = self._call_tokenizer
        else:
            self._impl = self._call_parser

    def __call__(self, text: str, lang: str | None = None):
        return self._impl(text, lang)

    def _call_agg_tokenizer(self, text: str, lang: str | None = None):
        assert lang is not None, "Expected 'lang' to be set for AggregateTokenizer."
        return self._tokenizer.text_to_ids(text, lang)

    def _call_tokenizer(self, text: str, lang: str | None = None):
        return self._tokenizer.text_to_ids(text)

    def _call_parser(self, text: str, lang: str | None = None):
        return self._tokenizer(text)


def _identity(x):
    return x
