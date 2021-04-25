NeMo Speaker Diarization Configuration Files
============================================

Since the speaker diarization model is not a fully-trainable end-to-end model, but an inference pipeline, we use **diarizer** 
instead of **model** which is used in other tasks.

The diarizer section requires information about the datasets being used, models used in this pipline, as well as inference related 
parameters such as postprocessing of each models. The sections on this page cover each of these in more detail.

The example configuration files for speaker diarization can be found in ``<NeMo_git_root>/examples/speaker_recognition/conf/speaker_diarization.yaml>``.

.. note::
  For more information about the models, configs, finetuning, tuning threshold, and evaluation, refer to ``<NeMo_git_root>/tutorials/speaker_recognition/Speaker_Diarization_Inference.ipynb``.
  For other applications such as possible integration with ASR, see ``<NeMo_git_root>/tutorials/speaker_recognition/ASR_with_SpeakerDiarization.ipynb``.


Dataset Configuration
---------------------

In contrast to other ASR related tasks or models in NeMo, speaker diarization supported in NeMo is a modular inference pipeline. Datasets here 
denotes the data you would like to perform speaker diarization on. An example Speaker Diarization dataset configuration looks similar
to the following:

.. code-block:: yaml

  diarizer:
    num_speakers: 2 # for each recording
    out_dir: ??? 
    paths2audio_files: null # either list of audio file paths or file containing paths to audio files for which we need to perform diarization.
    path2groundtruth_rttm_files: null # (Optional) either list of rttm file paths or file containing paths to rttm files (this can be passed if we need to calculate DER rate based on our ground truth rttm files).
    ...
    
.. note::
  We expect audio and the corresponding RTTM to have the same base name. The name should be unique.


Diarizer Architecture Configurations
------------------------------------

.. code-block:: yaml

  diarizer:
  ...
    vad:
      model_path: null #.nemo local model path or pretrained model name or none
      window_length_in_sec: 0.15
      shift_length_in_sec: 0.01
      threshold: 0.5 # tune threshold on dev set. Check <NeMo_git_root>/scripts/voice_activity_detection/vad_tune_threshold.py
      vad_decision_smoothing: True
      smoothing_params:
        method: "median" 
        overlap: 0.875

    speaker_embeddings:
      oracle_vad_manifest: null # leave it null if to perform diarization with above VAD model else path to manifest file genrerated as shown in Datasets section
      model_path: ??? #.nemo local model path or pretrained model name
      window_length_in_sec: 1.5
      shift_length_in_sec: 0.75
