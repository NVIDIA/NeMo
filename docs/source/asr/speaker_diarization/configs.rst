NeMo Speaker Diarization Configuration Files
============================================

Since speaker diarization model here is not a fully-trainble End-to-End model but an inference pipeline, we use **diarizer** instead of **model** which is used in other tasks.

The diarizer section will generally require information about the dataset(s) being
used, models used in this pipline, as well as inference related parameters such as postprocessing of each models.
The sections on this page cover each of these in more detail.


Example configuration files for speaker diarization can be found in the
``<NeMo_git_root>/examples/speaker_recognition/conf/speaker_diarization.yaml>``

.. note::
  For model details and deep understanding about configs, finetuning, tuning threshold, and evaluation, 
  please refer to ``<NeMo_git_root>/tutorials/speaker_recognition/Speaker_Diarization_Inference.ipynb``;
  for other applications such as possible integration with ASR, have a look at ``<NeMo_git_root>/tutorials/speaker_recognition/ASR_with_SpeakerDiarization.ipynb``.


Dataset Configuration
-----------------------

In contrast to other ASR related tasks or models in NeMo, speaker diarization supported in NeMo is a modular inference pipeline.
Datasets here denotes the data you would like to perform speaker diarization on. 

An example Speaker Diarization dataset configuration could look like:

.. code-block:: yaml

  diarizer:
    oracle_num_speakers: ??? # if number of speakers is known input it or pass null if not known. Accepts int or path to file containing uniq-id and num of speakers of that session
    max_num_speakers: 8 # max number of speakers for each recording. If oracle num speakers is passed this value is ignored
    out_dir: ??? 
    paths2audio_files: ??? # file containing paths to audio files for diarzation
    path2groundtruth_rttm_files: null # (Optional) file containing paths to rttm files (this can be passed if we need to calculate DER rate based on ground truth rttm files).
    ...
    
.. note::
  We expect audio and the corresponding RTTM to have the same base name and the name should be unique.


Diarizer Architecture Configurations
-------------------------------------

.. code-block:: yaml

  diarizer:
  ...
    vad:
      model_path: null #.nemo local model path or pretrained model name or none
      window_length_in_sec: 0.15
      shift_length_in_sec: 0.01
      vad_decision_smoothing: True
      smoothing_params:
        method: "median" 
        overlap: 0.875
      threshold: null # if not null, use this value and ignore the following onset and offset. Warning! it will be removed in release 1.5!
      postprocessing_params: # Tune parameter on dev set. Check <NeMo_git_root>/scripts/voice_activity_detection/vad_tune_threshold.py
        onset: 0.4
        offset: 0.7
        pad_onset: 0.05
        pad_offset: -0.1
        min_duration_on: 0.2 
        min_duration_off: 0.2    
        filter_speech_first: True 

    speaker_embeddings:
      oracle_vad_manifest: null # leave it null if to perform diarization with above VAD model else path to manifest file genrerated as shown in Datasets section
      model_path: ??? #.nemo local model path or pretrained model name
      window_length_in_sec: 1.5 # for better performace try with 3 sec and 1.5 sec
      shift_length_in_sec: 0.75 # for better performace try with 1.5 sec and 0.75 sec


