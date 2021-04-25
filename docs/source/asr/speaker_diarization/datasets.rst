Datasets
========

For steps on how to prepare datasets for training, validating VAD, and speaker embedding models, refer to the 
:doc:`Speech Classification Datasets <../speech_classification/datasets>` and :doc:`Speaker Recogniton Datasets <../speaker_recognition/datasets>` 
sections.

For Speaker Diarization inference, ``diarizer`` expects either list of paths to audio files or a file containing absolute paths to 
audio files. 

For generating a file that contains paths to audio files (which we call ``scp file``), use the ``find`` bash command as shown below:

.. code-block:: bash

  find $PWD/{data_dir} -iname '*.wav' > path_to_audiofiles.scp


Preparing the Evaluation Dataset
--------------------------------

To score with a diarizer model, we need to provide an scp like file as the groundtruth label file. Each groundtruth label file should 
be in NIST Rich Transcription Time Marked (RTTM) format. Take one line from a RTTM file, for example:

.. code-block:: bash

  SPEAKER TS3012d.Mix-Headset 1 331.573 0.671 <NA> <NA> MTD046ID <NA> <NA>


Prepraing an ORACLE Manifest
----------------------------

To perform an ORACLE diarization, that is taking speech activity timestamps from groundtruths instead of VAD output, ``diarizer`` 
expects an ``oracle_manifest`` file that contains paths to audio files with offsets for start time and duration of segment.

To prepare an ``oracle_manifest`` file, use the helper function from ``speaker_utils`` as shown below:

.. code-block:: python

    from nemo.collections.asr.parts.speaker_utils import write_rttm2manifest

    oracle_manifest = os.path.join(os.getcwd(),'oracle_manifest.json')
    write_rttm2manifest(paths2audio_files=paths2audio_files,
                    paths2rttm_files=path2groundtruth_rttm_files,
                    manifest_file=oracle_manifest)  

Here ``paths2audio_files`` and ``path2groundtruth_rttm_files`` are lists containing paths to audio files.
