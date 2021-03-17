Datasets
========
Check out page :doc:`Speech Classification Datasets <../sc/datasets>` and :doc:`Speaker Recogniton Datasets <../speaker_recognition/datasets>` 
for preparing datasets for training and validating VAD and speaker embedding models respectively.

Preparing Evaluation Dataset
-----------

The groundtruth label files should be in NIST Rich Transcription Time Marked (RTTM) format. Take one line from a rttm file for example.

.. code-block:: json

  SPEAKER TS3012d.Mix-Headset 1 331.573 0.671 <NA> <NA> MTD046ID <NA> <NA>