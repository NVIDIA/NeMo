Datasets
========

This page is about preparing input dataset for diarization inference. To train or fine-tune speaker diarization system, the speaker embedding extractor should be trained or fine-tuned. Check out page :doc:`Speech Classification Datasets <../speech_classification/datasets>` and :doc:`Speaker Recogniton Datasets <../speaker_recognition/datasets>` 
for preparing datasets for training and validating VAD and speaker embedding models respectively. 


Preparation of input data
-------------------------

diarization inference is based Hydra configurations which is fullfilled by .yaml files. See `NeMo Speaker Diarization Configuration Files <../configs>`_ for setting up the input configuration file for speaker diarization. Input data should be provided in line delimited JSON format as below:
	
.. code-block:: bash

  {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, label: "infer", "text": "-", "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath": "/path/to/uem/file"}

We refer to this file as manifest file and the path of the manifest file should be provided in Hydra configuration as follows:

.. code-block:: bash
   
	diarizer.manifest_filepath="path/to/manifest/input_manifest.json"

In the input manifest file, audio_filepath item is required argument.


.. note::
	We expect all the provided files (e.g. audio, rttm, text) to have the same base name and the name should be unique (uniq-id).

``audio_filepath`` (Required):
  - a file containing absolute paths to audio files or list of paths to audio files. 

``num_speakers`` (Optional):
  - If number of speakers is known, provide the integer number or assign null if not known. 
	
``rttm_filepath`` (Optional):
  - To evaluate a diarization system with known rttm files, One needs to provide Rich Transcription Time Marked (RTTM) files as groundtruth label files. If RTTM files are provided, the diarization evaluation will be initiated.
	Take one line from a RTTM file for example:

.. code-block:: bash

  SPEAKER TS3012d.Mix-Headset 1 331.573 0.671 <NA> <NA> MTD046ID <NA> <NA>

``text`` (Optional):
  - Ground truth transcription for diarization with ASR inference. Provide the ground truth transcription of the given audio file in string format

.. code-block:: bash

  {"text": "this is an example transcript"}

``uem_filepath`` (Optional):
  - UEM file is used for specifying the scoring regions to be evaluated in the given audio file.
	UEMfile follows the following convention: ``<uniq-id> <channel ID> <start time> <end time>``
	<channel ID> is set to 1.

  - Example line of UEM file:

.. code-block:: bash
  
    TS3012d.Mix-Headset 1 12.31 108.98
    TS3012d.Mix-Headset 1 214.00 857.09

``ctm_filepath`` (Optional):
  - CTM file is used for the evaluation of word-level diarization result and word-timestamp alignment.
	CTM file follows the following convention: ``<uniq-id> <speaker ID> <word start time> <word end time> <word> <confidence>``
	Since confidence is not required for evaluating diarization result, it can have any value. Note that the ``<speaker_id>`` should be exactly match with speaker IDs in RTTM. 

  - Example lines of CTM file:

.. code-block:: bash
  
   TS3012d.Mix-Headset MTD046ID 12.879 0.32 okay 0
   TS3012d.Mix-Headset MTD046ID 13.203 0.24 yeah 0


Evaluation on benchmark dataset
-------------------------------

The following instructions can help you to reproduce the expected diarization performance on two English dialogue datasets. The following results are evaluations based on 0.25 second collar without evaluating overlapped speech. The evaluation is based on oracle VAD results from RTTM files. Therefore, speaker error rate (SER) is equal to confusion error since oracle VAD has no miss detection or false alarm.

AMi Meeting Corpus
~~~~~~~~~~~~~~~~~~

The followings are the suggested parameters for reproducing the diarization performance for AMI test set.

.. code-block:: bash

  diarizer.manifest_filepath="/path/to/AMItest_input_manifest.json"
  diarizer.oracle_num_speakers=null # Performing unknown speaker case
  diarizer.oracle_vad=True # Use oracle VAD extracted from RTTM files.
  diarizer.collar=0.25
  diarizer.ignore_overlap=True 
  diarizer.speaker_embeddings.model_path = ``titanet_large`` 
  diarizer.speaker_embeddings.window_length_in_sec=[3,1.5,1.0,0.5] # Multiscale setting
  diarizer.speaker_embeddings.shift_length_in_sec=[1.5,0.75,0.5,0.25] # Multiscale setting 
  diarizer.speaker_embeddings.parameters.multiscale_weights=[0.4,0.3,0.2,0.1] # More weights on the longer scales

This setup is expected to reproduce speaker error rate  of 1.19% on AMI test set:

To evaluate the performance on AMI Meeting Corpus, the following instructions can help:
  - Download AMI Meeting Corpus from `AMI website <https://groups.inf.ed.ac.uk/ami/corpus/>`_.
  - Get the test set (whitelist) from `Pyannotate AMI testset whitelist <https://raw.githubusercontent.com/pyannote/pyannote-audio/master/tutorials/data_preparation/AMI/MixHeadset.test.lst>`_.
  - The merged RTTM file for AMI testset can be downloaded from `Pyannotate AMI testset RTTM file <https://raw.githubusercontent.com/pyannote/pyannote-audio/master/tutorials/data_preparation/AMI/MixHeadset.test.rttm>`_. Note that this file should be split into individual rttm files. Download split rttm files for AMI testset from `AMI testset split RTTM files <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/AMI_corpus/test/split_rttms.tar.gz>`_.
  - Generate an input manifest file using ``<NeMo_git_root>/scripts/speaker_tasks/pathsfiles_to_manifest.py``


CallHome American English Speech (CHAES), LDC97S42: 2-speaker subset (CH109)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CH109 is a subset of CHAES dataset which has only two speakers in one session. 
The followings are the suggested parameters for reproducing the diarization performance for CH109 set.

.. code-block:: bash

  diarizer.manifest_filepath="/path/to/ch109_input_manifest.json"
  diarizer.oracle_num_speakers=2 (Since there are exactly 2 speakers per each CH109 session)
  diarizer.oracle_vad=True # Use oracle VAD extracted from RTTM files.
  diarizer.collar=0.25
  diarizer.ignore_overlap=True 
  diarizer.speaker_embeddings.model_path = ``titanet_large`` 
  diarizer.speaker_embeddings.window_length_in_sec=[1.5,1.0,0.5] # Multiscale setting
  diarizer.speaker_embeddings.shift_length_in_sec=[0.75,0.5,0.25] # Multiscale setting
  diarizer.speaker_embeddings.parameters.multiscale_weights=[0.33,0.33,0.33] # Equal weights

This setup is expected to reproduce confusion error of 0.82% on CH109 set:

To evaluate the performance on AMI Meeting Corpus, the following instructions can help:
  - Download CHAES Meeting Corpus at LDC website `LDC97S42 <https://catalog.ldc.upenn.edu/LDC97S42>`_ (CHAES is not publicly available).
  - Get the CH109 filelist (whitelist) from `CH109 whitelist <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/CH109/ch109_whitelist.txt>`_.
  - Download RTTM files for CH109 set from `CH109 RTTM files <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/CH109/split_rttms.tar.gz>`_.
  - Generate an input manifest file using ``<NeMo_git_root>/scripts/speaker_tasks/pathsfiles_to_manifest.py``

