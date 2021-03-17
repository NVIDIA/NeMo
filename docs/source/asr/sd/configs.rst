NeMo Speaker Diarization Configuration Files
============================

This page covers NeMo configuration file setup that is specific to Speaker Diarization (Inference) task.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../introduction/core` page.

Since speaker diarization model here is not a fully-trainble End-to-End model but a inference pipeline, we use **diarizer** instead of **model** which is used in other tasks.

The diarizer section will generally require information about the dataset(s) being
used, models used in this pipline , as well as inference ratled parameters such as postprocessing of each models.
The sections on this page cover each of these in more detail.


Example configuration files for speaker diarization can be found in the
``<NeMo_git_root>/examples/speaker_recognition/conf/speaker_diarization.yaml>``

.. note::
  For model details and deep understanding about configs, finetuning, tuning threshold, and evaluation, 
  please refer to ``<NeMo_git_root>/tutorials/speaker_recognition/Speaker_Diarization_Inference.ipynb``;
  for other applications such as possible integration with ASR, have a look at ``<NeMo_git_root>/tutorials/speaker_recognition/ASR_with_SpeakerDiarization.ipynb``.


Dataset Configuration
---------------------

In contrast to other ASR related tasks or models, speaker diarization supported in NeMo now is a modular inference pipeline.
Datasets here denotes the data you would like to perform speaker diarization on. 

An example Speaker Diarization dataset configuration could look like:

.. code-block:: yaml

  diarizer:
    num_speakers: 2 # for each recording
    out_dir: ??? 
    paths2audio_files: null # either list of audio file paths or file containing paths to audio files for which we need to perform diarization.
    path2groundtruth_rttm_files: null: # (Optional) either list of rttm file paths or file containing paths to rttm files (this can be passed if we need to calculate DER rate based on our ground truth rttm files).
    ...
    
.. note::
  Note we expect audio and corresponding RTTM have same base name and the name should be unique.


Diarizer Architecture Configurations
---------------------------------

.. code-block:: yaml

  diarizer:
  ...
    vad:
      model_path: null #.nemo local model path or pretrained model name or none
      window_length_in_sec: 0.15
      shift_length_in_sec: 0.01
      threshold: 0.5 # tune threshold on dev set. Check scripts/vad_tune_threshold.py
      vad_decision_smoothing: True
      smoothing_params:
        method: "median" 
        overlap: 0.875

    speaker_embeddings:
      oracle_vad_manifest: null
      model_path: ??? #.nemo local model path or pretrained model name
      window_length_in_sec: 1.5
      shift_length_in_sec: 0.75


