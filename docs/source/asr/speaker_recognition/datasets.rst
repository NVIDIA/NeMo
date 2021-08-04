Datasets
========

.. _HI-MIA:

HI-MIA
--------

Run the script to download and process ``hi-mia`` dataset in order to generate files in the supported format of  ``nemo_asr``. You should set the data folder of 
hi-mia using ``--data_root``. These scripts are present in ``<nemo_root>/scripts``

.. code-block:: bash

    python get_hi-mia_data.py --data_root=<data directory> 

After download and conversion, your `data` folder should contain directories with following set of files as:

* `data/<set>/train.json`
* `data/<set>/dev.json` 
* `data/<set>/{set}_all.json` 
* `data/<set>/utt2spk`


All-other Datasets
------------------

These methods can be applied to any dataset to get similar training manifest files.

First we prepare scp file(s) containing absolute paths to all the wav files required for each of the train, dev, and test set. This can be easily prepared by using ``find`` bash command as follows:

.. code-block:: bash 

    !find {data_dir}/{train_dir}  -iname "*.wav" > data/train_all.scp
    !head -n 3 data/train_all.scp


Since we created the scp file for the train, we use `scp_to_manifest.py` to convert this scp file to a manifest file and then optionally split the files to train \& dev for evaluating the models while training by using the --split flag. 
We wont be needing the --split option for the test folder. Accordingly please mention the id number, which is the field num separated by / to be considered as the speaker label.
After the download and conversion, your data folder should contain directories with manifest files as:
    
* `data/<path>/train.json`
* `data/<path>/dev.json`
* `data/<path>/train_all.json`
    
Each line in the manifest file describes a training sample - audio_filepath contains the path to the wav file, duration it's duration in seconds, and label is the speaker class label:

.. code-block:: json
    
    {"audio_filepath": "<absolute path to dataset>/audio_file.wav", "duration": 3.9, "label": "speaker_id"}


Tarred Datasets
---------------

Similarly to ASR, you can tar your audio files and use ASR Dataset class ``TarredAudioToSpeechLabelDataset`` (corresponding to the ``AudioToSpeechLabelDataset``) for this case.

If you want to use tarred dataset, have a look at `ASR Tarred Datasets <../datasets.html#tarred-datasets>`__.