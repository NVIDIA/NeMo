Datasets
========
Check out page :doc:`Speech Classification Datasets <../speech_classification/datasets>` and :doc:`Speaker Recogniton Datasets <../speaker_recognition/datasets>` 
for preparing datasets for training and validating VAD and speaker embedding models respectively.

For Speaker Diarization inference, ``diarizer`` expects following arguments :

  - paths2audio_files
  - oracle_num_speakers
  - path2groundtruth_rttm_files [Optional]
  - oracle_vad_manifest (if vad/rttm files are known)[Optional but suggested for good performace. This avoids FPR and TPR]

paths2audio_files:
  - a file containing absolute paths to audio files or list of paths to audio files. 
  For generating a file that contains paths to audio files, you can simply use ``find`` bash command as shown below:
.. code-block:: bash

  find $PWD/{data_dir} -iname '*.wav' > path_to_audiofiles.txt

.. note::
  We expect audio and the corresponding RTTM files to have the same base name and the name should be unique (uniq-id).

oracle_num_speakers:
  - if number of speakers is known input it or pass null if not known. Accepts int or path to file containing uniq-id with num of speakers of that session 

  Sample line (uniq-id `iaaa` session has 2 unique speakers )::
    
    iaaa 2

path2groundtruth_rttm_files [Optional]:
  - To evaluate the diarizer system with known rttm files, One needs to provide a txt file (path2groundtruth_rttm_files) for groundtruth label files.
  use above mentioned ``find`` command to get all reference rttm files (use '\*.rttm' as search pattern)

Each groundtruth label file should be in NIST Rich Transcription Time Marked (RTTM) format. Take one line from a RTTM file for example:

.. code-block:: bash

  SPEAKER TS3012d.Mix-Headset 1 331.573 0.671 <NA> <NA> MTD046ID <NA> <NA>


oracle_vad_manifest [Optional but suggested for good performace. This reduces FPR and TPR]:
  - To perform just oracle diarization, that is taking speech activity time stamps from groundtruth RTTMs or another VAD from wild instead of NeMo VAD output, ``diarizer`` expects an oracle manifest json file that contains paths to audio files with offset for start time and duration of segment.

To prepare an oracle manifest file, use the script from ``scripts`` folder as shown below:

.. code-block:: bash

  python $NeMo/scripts/ --paths2rttm_files=<path2groundtruth_rttm_files.txt> --paths2audio_files=<paths2audio_files.txt> --manifest_file=<output_oracle_manifest_file.json>

Here ``paths2audio_files`` and ``path2groundtruth_rttm_files`` are files containing paths to audio files as shown above.

AMI Meeting Corpus
------------------

The following are the suggested parameters for getting Speaker Error Rate (SER) of 2.13% on AMI Lapel test set corpus:
  - diarizer.oracle_num_speakers = null (performing unknown speaker case)
  - diarizer.speaker_embeddings.model_path = ``ecapa_tdnn`` (This model is trained on voxceleb dataset. ``Use this model for simialr non-telephonic speech datasets``)
  - diarizer.speaker_embeddings.window_length_in_sec = 1.5
  - diarizer.speaker_embeddings.shift_length_in_sec = 0.75 

Input paths2audio_files, paths2rttm_files and oracle_vad_manifest by following steps as shown above

CallHome LDC97S42 (CH109)
-------------------------

The following are the suggested parameters for getting Speaker Error Rate (SER) of 1.19% on CH109 set:
  - diarizer.oracle_num_speakers = 2 (since there are exactly 2 speakers per each ch109 session)
  - diarizer.speaker_embeddings.model_path = ``ecapa_tdnn`` (This model is trained on voxceleb and telephonic speech Fisher and SWBD. ``Use this model for similar telephonic speech datasets``)
  - diarizer.speaker_embeddings.window_length_in_sec = 1.5
  - diarizer.speaker_embeddings.shift_length_in_sec = 0.75

Input paths2audio_files, paths2rttm_files and oracle_vad_manifest by following steps as shown above
