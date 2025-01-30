dataset_meta_info = {
    'vctk': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths.json',
        'audio_dir' : '/datap/misc/Datasets/VCTK-Corpus',
        'feature_dir' : '/datap/misc/Datasets/VCTK-Corpus',
    },
    'riva_challenging': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/challengingLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json',
        'audio_dir' : '/datap/misc/Datasets/riva',
        'feature_dir' : '/datap/misc/Datasets/riva',
    },
    'riva_challenging_shehzeen': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/challengingLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths_v2.json',
        'audio_dir' : '/Data/RivaData/riva',
        'feature_dir' : '/Data/RivaData/riva',
    },
    'riva_challenging_nozeros': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/riva_challenging_nozeros.json',
        'audio_dir' : '/datap/misc/Datasets/riva',
        'feature_dir' : '/datap/misc/Datasets/riva',
    },
    'libri_val': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/libri360_val.json',
        'audio_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
        'feature_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
    },
    'libri_val_shehzeen': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/libri360_val.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
    },
    'libri_unseen_test': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
        'feature_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
    },
    'libri_unseen_test_shehzeen_phoneme': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
        'tokenizer_names': ['english_phoneme'],
    },
    'libri_unseen_test_shehzeen_sep_char': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
        'tokenizer_names': ['english_chartokenizer'],
    },
    'libri_unseen_test_shehzeen_shared_char': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
        'tokenizer_names': ['chartokenizer'],
    },
    'libri_unseen_test_shehzeen_sp': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
        'tokenizer_names': ['multilingual_sentencepiece'],
    },
    'libri_unseen_test_shehzeen': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/test_clean_withContextAudioPaths.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
    },
    'libri_seen_test_v2': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/libri_seen_evalset_from_testclean_v2.json',
        'audio_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
        'feature_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
    },
    'libri_seen_test_v2_shehzeen': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/libri_seen_evalset_from_testclean_v2.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
    },
    'libri_unseen_val': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/dev_clean_withContextAudioPaths_evalset.json',
        'audio_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
        'feature_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
    },
    'spanish_cml_phoneme': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es'
    },
    'spanish_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['spanish_chartokenizer'],
        'whisper_language': 'es'
    },
    'spanish_cml_shared_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['chartokenizer'],
        'whisper_language': 'es'
    },
    'spanish_cml_sp': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'es'
    },
    'german_cml_phoneme': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_german_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'german_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_german_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['german_chartokenizer'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'german_cml_shared_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_german_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['chartokenizer'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'german_cml_sp': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_german_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'de',
        'load_cached_codes_if_available': False
    },
    'french_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_french_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['french_chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'french_cml_shared_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_french_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['chartokenizer'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'french_cml_sp': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_french_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'fr',
        'load_cached_codes_if_available': False
    },
    'italian_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_italian_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['italian_chartokenizer'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    },
    'italian_cml_shared_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_italian_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['chartokenizer'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    },
    'italian_cml_sp': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_italian_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'it',
        'load_cached_codes_if_available': False
    },
    'dutch_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_dutch_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'tokenizer_names': ['dutch_chartokenizer'],
        'whisper_language': 'nl',
        'load_cached_codes_if_available': False
    },
    'dutch_cml_shared_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_dutch_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'tokenizer_names': ['chartokenizer'],
        'whisper_language': 'nl',
        'load_cached_codes_if_available': False
    },
    'dutch_cml_sp': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_dutch_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'nl',
        'load_cached_codes_if_available': False
    },
    'portuguese_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_portuguese_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_portuguese_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_portuguese_v0.1',
        'tokenizer_names': ['portuguese_chartokenizer'],
        'whisper_language': 'pt',
        'load_cached_codes_if_available': False
    },
    'polish_cml_sep_char': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_polish_v0.1/test_withAudioCodes_codec21Khz_no_eliz_filtered_100subset.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_polish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_polish_v0.1',
        'tokenizer_names': ['polish_chartokenizer'],
        'whisper_language': 'pl',
        'load_cached_codes_if_available': False
    },
}