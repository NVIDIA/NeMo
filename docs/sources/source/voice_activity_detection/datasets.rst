Datasets
========

.. _Freesound_dataset:

Freesound database
-----------------------------------

`Freesound <http://www.freesound.org/>`_ is a website that aims to create a huge open collaborative database of audio snippets, samples, recordings, bleeps. Audio samples are released under Creative Commons licenses that allow their reuse. Researchers and developers can access Freesound content using the Freesound API to retrieve meaningful sound information such as metadata, analysis files and the sounds themselves. (see :cite:`vad-dataset-font2013freesound`)

We use freesound "background categories" as background/non-speech audio samples for voice activity detection.

.. _GoogleSpeechCommands_dataset_vad:

Google Speech Commands Dataset
-----------------------------------

Google released two versions of Speech Commands dataset (see :cite:`vad-dataset-warden2018speech`), which contains short audio clips of a fixed number of command words such as “stop”, “go”, “up”, “down”, etc spoken by a large number of speakers. The first version containing 65k samples over 30 classes and the second containing 110k samples over 35 classes.
We refer to these datasets as v1 and v2, and currently we use v2 data as speech audio utterance for voice activity detection.

.. _Freesound_download_resamplet:

Freesound Download and Resample
-----------------------------------

Please have a look at Data Preparation section in Tutorial.

References
----------

.. bibliography:: vad_all.bib
    :style: plain
    :labelprefix: VAD-DATASET
    :keyprefix: vad-dataset-