NeMo Speaker Diarization API
=============================


Model Classes
-------------
.. autoclass:: nemo.collections.asr.models.ClusteringDiarizer
    :show-inheritance:
    :members: init_speaker_model, set_vad_model, init_vad_model, setup_vad_test_data, setup_spkr_test_data, run_vad, extract_embeddings, path2audio_files_to_manifest, diarize, make_nemo_file_from_folder, save_to, unpack_nemo_file, restore_from

Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.mixins.DiarizationMixin
    :show-inheritance:
    :members:

Miscellaneous Classes
---------------------

Diarization with ASR
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.utils.diarization_utils.ASR_DIAR_OFFLINE
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.utils.decoder_ts_utils.ASR_TIMESTAMPS
    :show-inheritance:
    :members:

