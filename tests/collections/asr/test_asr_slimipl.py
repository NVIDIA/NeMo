
import os
import tempfile
import pytest
import torch
import re
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecCTCModelBPE,EncDecHybridRNNTCTCBPEModel
import json
from unittest.mock import mock_open, patch
from nemo.collections.asr.parts.utils.ipl_utils import (
    process_manifest, 
    write_cache_manifest,
    handle_multiple_tarr_filepaths 
)

@pytest.fixture(scope="module")
def fast_conformer_ctc_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large")
    return model

@pytest.fixture(scope="module")
def fast_conformer_hybrid_model():
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="stt_en_fastconformer_hybrid_large_pc")
    return model


class TestPseudoLabelGeneration:
    @pytest.mark.unit
    def test_generate_pseudo_labels(self, test_data_dir, fast_conformer_ctc_model, fast_conformer_hybrid_model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        asr_model_ctc = fast_conformer_ctc_model.to(device)
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/an4_val.json'))
        texts = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as m:
                for ix, line in enumerate(m):

                    line = line.replace("tests/data/", "tests/.data/").replace("\n", "")
                    data = json.loads(line)
                    texts.append(data['text'])
                    f.write(f"{line}\n")

            f.seek(0)
            hypotheses = asr_model_ctc.generate_pseudo_labels_ctc(cache_manifest=f.name)
            assert len(hypotheses) == len(texts)
            assert hypotheses[0] == texts[0]

            asr_model_hybrid = fast_conformer_hybrid_model
            hypotheses = asr_model_hybrid.generate_pseudo_labels_hybrid(cache_manifest=f.name)
            assert len(hypotheses) == len(texts)
            assert hypotheses[0].lower().replace(',', '') == texts[0]
    
    @pytest.mark.unit
    def test_generate_pseudo_labels_tar(self, test_data_dir, fast_conformer_ctc_model, fast_conformer_hybrid_model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        asr_model_ctc = fast_conformer_ctc_model.to(device)
        tarred_path = os.path.abspath(os.path.join(test_data_dir, 'asr/libri_tarred_test/'))
        sharded_manifests = os.path.abspath(os.path.join(tarred_path, 'sharded_manifests/manifest__OP_0..1_CL_.json'))
        tarred_audio_filepaths = os.path.abspath(os.path.join(tarred_path, 'audio__OP_0..1_CL_.tar'))
        manifest_data = process_manifest(os.path.abspath(os.path.join(tarred_path, 'sharded_manifests/manifest_0.json')))
        asr_model_ctc.cfg.train_ds.is_tarred=True
        hypotheses = asr_model_ctc.generate_pseudo_labels_ctc(cache_manifest=sharded_manifests, 
                                                              tarred_audio_filepaths = tarred_audio_filepaths)
        assert len(hypotheses) == 10
        assert hypotheses[0] == manifest_data[0]["text"]
        asr_model_hybrid = fast_conformer_hybrid_model
        asr_model_hybrid.cfg.train_ds.is_tarred=True
        hypotheses = asr_model_hybrid.generate_pseudo_labels_hybrid(cache_manifest=sharded_manifests, 
                                                              tarred_audio_filepaths = tarred_audio_filepaths)
        assert len(hypotheses) == 10
        assert re.sub(r'[.,?]', '', hypotheses[0]).lower() == manifest_data[0]["text"]

    @pytest.mark.unit
    @patch('builtins.open',new_callable=mock_open)
    def test_write_whole_cache(self, mock_open):
        cache_manifest = 'test_cache.json'
        hypotheses = [['test1', 'test2'], ['test3', 'test4']]
        data = [
            [{'audio_filepath': "audio_0.wav", 'duration': '10', 'text': ''},
             {'audio_filepath': "audio_1.wav", 'duration': '12', 'text': ''}],
            [{'audio_filepath': "audio_2.wav", 'duration': '14', 'text': ''},
             {'audio_filepath': "audio_3.wav", 'duration': '16', 'text': ''}]
        ]
        update_whole_cache = True

        write_cache_manifest(cache_manifest, hypotheses, data, update_whole_cache)

        mock_open.assert_called_once_with(cache_manifest, 'w', encoding='utf-8')
        handle = mock_open()
        write_calls = handle.write.call_args_list
        expected_data = (
            '{"audio_filepath": "audio_0.wav", "duration": "10", "text": "test1"}\n'
            '{"audio_filepath": "audio_1.wav", "duration": "12", "text": "test2"}\n'
            '{"audio_filepath": "audio_2.wav", "duration": "14", "text": "test3"}\n'
            '{"audio_filepath": "audio_3.wav", "duration": "16", "text": "test4"}\n'
        )

        written_data = ''.join(call_arg.args[0] for call_arg in write_calls)
        assert written_data == expected_data
     
    @pytest.mark.unit    
    @patch(
        'builtins.open', 
        new_callable=mock_open, 
        read_data=(
            '{"audio_filepath": "audio_3.wav", "duration": "18", "text": "test1"}\n'
            '{"audio_filepath": "audio_4.wav", "duration": "10", "text": "test2"}\n'
            '{"audio_filepath": "audio_0.wav", "duration": "12", "text": "test1"}\n'
            '{"audio_filepath": "audio_1.wav", "duration": "14", "text": "test2"}'
        )
    )  
    @patch('random.shuffle', lambda x: x) 
    def test_write_partial_cache(self, mock_open, ):


        cache_manifest = 'test_cache.json'
        hypotheses = [["", ""]]
        data = [
            [{'audio_filepath': "audio_0.wav", 'duration': '12', 'text': 'test1'},
             {'audio_filepath': "audio_1.wav", 'duration': '14', 'text': 'test2'}]
        ]

        write_cache_manifest(cache_manifest, hypotheses, data, update_whole_cache = False)
        handle = mock_open()
        write_calls = handle.write.call_args_list
        expected_data = (
            '{"audio_filepath": "audio_3.wav", "duration": "18", "text": "test1"}\n'
            '{"audio_filepath": "audio_4.wav", "duration": "10", "text": "test2"}\n'
            '{"audio_filepath": "audio_0.wav", "duration": "12", "text": ""}\n'
            '{"audio_filepath": "audio_1.wav", "duration": "14", "text": ""}\n'
        )

        written_data = ''.join(call_arg.args[0] for call_arg in write_calls)

        assert written_data == expected_data
    

    @pytest.mark.unit    
    def test_handle_multiple_tarr_filepaths(self):
        manifest_file = 'cache_manifests_0.json'
        tmpdir = '/tmp'
        number_of_manifests = 3
        tarr_file = '/data/audio_0.tar'
        expected_temporary_manifest = '/tmp/temp_cache_manifests_{0..2}.json'
        expected_expanded_audio_path = '/data/audio_{0..2}.tar'

        result = handle_multiple_tarr_filepaths(manifest_file, tmpdir, number_of_manifests, tarr_file)
        assert result == (expected_temporary_manifest, expected_expanded_audio_path)

