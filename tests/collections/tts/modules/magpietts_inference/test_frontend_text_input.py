# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from examples.tts.magpietts_inference import create_argument_parser

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import _get_record_texts
from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
    EasyMagpieMultiturnUserAudioInferenceConfig,
    _build_phoneme_frontend,
    _frontend_text_input,
)
from nemo.collections.tts.modules.magpietts_inference.utils import _build_easy_magpie_config, _select_runner_cls
from nemo.collections.tts.parts.utils.tts_dataset_utils import tokenize_text_with_phoneme_spans

pytestmark = pytest.mark.unit
CLI = ['--codecmodel_path', '/codec.nemo', '--datasets_json_path', '/datasets.json', '--out_dir', '/output']


def _runner(enabled):
    runner = object.__new__(EasyMagpieInferenceRunner)
    runner.config = EasyMagpieInferenceConfig(phoneme_frontend_as_text_input=enabled, batch_size=2)
    tokenizer = SimpleNamespace(
        tokens={'a': 4, 'b': 5, 'c': 6, '<pad>': 0},
        pad=0,
        bos_token_id=100,
        eos_token_id=101,
        unk_token_id=102,
        encode=Mock(side_effect=AssertionError('Shared IDs must not be encoded again')),
    )
    runner.model = SimpleNamespace(
        phoneme_tokenizer=tokenizer,
        pad_id=0,
        eos_id=999,
        text_phoneme_token_offset=1000,
        text_phoneme_vocab_size=103,
        enable_phoneme_text_input=True,
        phoneme_text_bop_marker='<bop>',
        phoneme_text_eop_marker='<eop>',
    )
    source_tokenizer = SimpleNamespace(**vars(tokenizer))
    runner.phoneme_frontend = SimpleNamespace(
        model=SimpleNamespace(phoneme_tokenizer=source_tokenizer),
        metadata={'checkpoint_sha256': 'fixture'},
        generate=Mock(
            side_effect=[
                SimpleNamespace(ipa='wʌn', token_ids=(0, 4, 5, 0), decoder_steps=12),
                SimpleNamespace(ipa='tu', token_ids=(6,), decoder_steps=9),
            ]
        ),
    )
    runner._current_precomputed_audio_embedding = None
    runner._batch_to_cuda = lambda batch: batch
    return runner


def _batch():
    return dict(
        text=torch.tensor([[11, 12, 999], [21, 999, 0]], dtype=torch.int32),
        text_lens=torch.tensor([3, 2]),
        raw_texts=['$1', '2'],
        languages=['en', 'en-US'],
        normalized_text=['one dollar', 'two'],
        phoneme_tokens=torch.tensor([[888, 888], [777, 777]]),
        phoneme_tokens_lens=torch.tensor([2, 2]),
    )


@pytest.mark.parametrize('mode', ['single_turn', 'multiturn_user_audio'])
def test_cli_defaults_off_and_accepts_frontend_text_input(mode):
    parser = create_argument_parser()
    args = parser.parse_args(CLI + ['--model_type', 'easy_magpie', '--easy_magpie_inference_mode', mode])
    assert args.phoneme_frontend_as_text_input is False
    assert _build_easy_magpie_config(args).phoneme_frontend_as_text_input is False
    args = parser.parse_args(
        CLI
        + [
            '--model_type',
            'easy_magpie',
            '--easy_magpie_inference_mode',
            mode,
            '--phoneme_frontend_nemo_file',
            '/frontend.nemo',
            '--phoneme_frontend_as_text_input',
        ]
    )
    assert _build_easy_magpie_config(args).phoneme_frontend_as_text_input is True


@pytest.mark.parametrize(
    'options,match',
    [
        ([], 'requires --phoneme_frontend_nemo_file'),
        (['--phoneme_frontend_nemo_file', '/frontend.nemo', '--dropout_text_input'], 'dropout_text_input'),
    ],
)
def test_cli_rejects_invalid_text_input_combinations(options, match):
    args = create_argument_parser().parse_args(CLI + ['--phoneme_frontend_as_text_input'] + options)
    with pytest.raises(ValueError, match=match):
        _build_easy_magpie_config(args)
    args.model_type = 'magpie'
    with pytest.raises(ValueError, match='easy_magpie'):
        _select_runner_cls(args)


@pytest.mark.parametrize(
    'model,frontend,match',
    [
        (SimpleNamespace(), None, 'requires a phoneme frontend'),
        (SimpleNamespace(enable_phoneme_text_input=False), '/f.nemo', 'enable_phoneme_text_input=True'),
        (SimpleNamespace(enable_phoneme_text_input=True), '/f.nemo', 'text_phoneme_token_offset'),
    ],
)
def test_unsupported_checkpoint_fails_before_loading_frontend(model, frontend, match):
    config = EasyMagpieInferenceConfig(phoneme_frontend_as_text_input=True, phoneme_frontend_nemo_file=frontend)
    with pytest.raises(ValueError, match=match):
        _build_phoneme_frontend(model, config)


@pytest.mark.parametrize('config_type', [EasyMagpieInferenceConfig, EasyMagpieMultiturnUserAudioInferenceConfig])
def test_text_input_mode_has_a_distinct_output_identifier(config_type):
    config = config_type(phoneme_frontend_nemo_file='/frontend.nemo')
    original = config.build_identifier()
    config.phoneme_frontend_as_text_input = True
    assert config.build_identifier() == original + '_IPATextInput'
    config.phoneme_frontend_as_text_input = False
    assert config.build_identifier() == original


def test_predicted_ids_replace_bpe_and_use_text_padding_and_eos():
    runner = _runner(True)
    batch = _batch()
    records = runner._set_frontend_phoneme_inputs(batch)
    assert batch['text'].dtype == torch.int32
    assert batch['text'].tolist() == [[1004, 1005, 999], [1006, 999, 0]]
    assert batch['text_lens'].tolist() == [3, 2]
    assert batch['phoneme_tokens'].tolist() == [[100, 4, 5, 101], [100, 6, 101, 0]]
    assert batch['phoneme_tokens_lens'].tolist() == [4, 3]
    assert batch['raw_texts'] == ['$1', '2']
    assert batch['normalized_text'] == ['one dollar', 'two']
    assert [call.args for call in runner.phoneme_frontend.generate.call_args_list] == [('$1', 'en'), ('2', 'en-US')]
    assert [r['tts_text_input'] for r in records] == ['<bop>wʌn<eop>', '<bop>tu<eop>']
    assert [r['tts_text_input_token_ids'] for r in records] == [[1004, 1005, 999], [1006, 999]]
    assert all(r['phoneme_token_transfer'] == 'direct_ids' for r in records)
    runner.model.phoneme_tokenizer.encode.assert_not_called()


def test_disabled_mode_keeps_bpe_inputs_exactly():
    runner = _runner(False)
    batch = _batch()
    text, lengths = batch['text'], batch['text_lens']
    records = runner._set_frontend_phoneme_inputs(batch)
    assert batch['text'] is text and batch['text_lens'] is lengths
    assert all('tts_text_input_token_ids' not in row for row in records)
    assert batch['phoneme_tokens'].tolist() == [[100, 4, 5, 101], [100, 6, 101, 0]]


def test_mapping_matches_existing_phoneme_text_tokenizer_path():
    runner = _runner(True)
    model = runner.model
    model.phoneme_text_bop_marker, model.phoneme_text_eop_marker = '[IPA]', '[/IPA]'
    output = SimpleNamespace(ipa='wʌn')
    record = _frontend_text_input(model, output, [100, 4, 5, 101])
    text_tokenizer = SimpleNamespace(encode=Mock(side_effect=AssertionError('No BPE encoding expected')))
    model.phoneme_tokenizer.encode.side_effect = None
    model.phoneme_tokenizer.encode.return_value = [4, 5]
    expected = tokenize_text_with_phoneme_spans(
        text_tokenizer=text_tokenizer,
        text_str=record['tts_text_input'],
        tokenizer_name='text',
        enable_phoneme_text_input=True,
        phoneme_tokenizer=model.phoneme_tokenizer,
        text_phoneme_token_offset=model.text_phoneme_token_offset,
        bop_marker=model.phoneme_text_bop_marker,
        eop_marker=model.phoneme_text_eop_marker,
    ) + [model.eos_id]
    assert record['tts_text_input_token_ids'] == expected == [1004, 1005, 999]
    text_tokenizer.encode.assert_not_called()


@pytest.mark.parametrize('ids', [[100, -1, 101], [100, 103, 101], [100, 101]])
def test_invalid_ipa_ids_are_not_used_as_text(ids):
    with pytest.raises(ValueError, match='phoneme'):
        _frontend_text_input(_runner(True).model, SimpleNamespace(ipa='wʌn'), ids)


def test_single_turn_logs_ipa_input_and_preserves_normalized_scoring_reference(tmp_path):
    runner = _runner(True)
    records = [dict(id='a', text='$1', normalized_text='one dollar'), dict(id='b', text='2', normalized_text='two')]

    class Dataset(torch.utils.data.Dataset):
        data_samples = [SimpleNamespace(text=row['text']) for row in records]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        def collate_fn(self, indices):
            assert indices == [0, 1]
            return _batch()

    seen = []

    def infer(batch, **kwargs):
        seen.append(batch['text'].tolist())
        return SimpleNamespace(
            predicted_audio=torch.zeros(2, 8),
            predicted_audio_lens=torch.tensor([8, 8]),
            predicted_codes=torch.zeros(2, 1, 1),
            predicted_codes_lens=torch.tensor([1, 1]),
            rtf_metrics={},
        )

    runner.model.infer_batch, runner.model.sample_rate = infer, 22050
    runner._run_decoder_only_inference(Dataset(), str(tmp_path), records, str(tmp_path), False, False)
    logged = [
        json.loads(line) for line in (tmp_path / 'singleturn_evaluation_manifest.jsonl').read_text().splitlines()
    ]
    assert seen == [[[1004, 1005, 999], [1006, 999, 0]]]
    assert [row['text'] for row in logged] == ['<bop>wʌn<eop>', '<bop>tu<eop>']
    assert [row['original_text'] for row in logged] == ['$1', '2']
    assert [row['normalized_text'] for row in logged] == ['one dollar', 'two']
    for row in logged:
        actual, normalized, reference = _get_record_texts(row)
        assert actual == row['text'] and normalized == reference == row['normalized_text']
