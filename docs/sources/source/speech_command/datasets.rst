Datasets
========

.. _GoogleSpeechCommands_dataset:


Google Speech Commands Dataset
-----------------------------------

The ability to recognize spoken commands with high accuracy can be useful in a variety of contexts.
To this end, Google released the Speech Commands dataset (see :cite:`speech-recognition-dataset-warden2018speech`,
which contains short audio clips of a fixed number of command words such as “stop”, “go”, “up”, “down”, etc spoken by a large number of speakers.
To promote the use of the set, Google also hosted a Kaggle competition, in which the winning team attained a multi-class accuracy of 91%.

We experimented with applying NeMo’s ASR classification models on mel spectrogram of the audio clips and found that they worked surprisingly well.
Adding data augmentation further improved the results.

Dataset
-----------------------------------

Google released two versions of the dataset with the first version containing 65k samples over 30 classes and the second containing 110k samples over 35 classes.
We refer to these datasets as v1 and v2, and currently we have metrics for v1 version in order to compare to the different metrics used by other papers.

Run the script to process Google Speech Commands dataset in order to generate files in the supported format of  `nemo_asr`.
You should set the data folder of Speech Commands using `--data_root` and the version of the dataset using `--data_version` as an int.

You can further rebalance the train set by passing the `--rebalance` flag.

.. code-block:: bash

    python process_speech_commands_data.py --data_root=<data directory> --data_version=<1 or 2> {--rebalance}

Then, you should have `train_manifest.json`, `validation_manifest.json` and `test_manifest.json`
in the directory `{data_root}/google_speech_recognition_v{1/2}`.

References
----------

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-DATASET
    :keyprefix: speech-recognition-dataset-