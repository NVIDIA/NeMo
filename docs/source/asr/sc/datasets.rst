Datasets
========

NeMo has scripts to convert several common ASR datasets into the format expected by the `nemo_asr` collection.
You can get started with those datasets by following the instructions to run those scripts in the section appropriate to each dataset below.

If you have your own data and want to preprocess it to use with NeMo ASR models, check out the `Preparing Custom ASR Data`_ section at the bottom of the page.

.. _freesound-dataset:
Freesound
-----------

`Freesound <http://www.freesound.org/>`_ is a website that aims to create a huge open collaborative database of audio snippets, samples, recordings, bleeps. 
Audio samples are released under Creative Commons licenses that allow their reuse. 
Researchers and developers can access Freesound content using the Freesound API to retrieve meaningful sound information such as metadata, analysis files and the sounds themselves. 

**Instructions**

Go to `NeMo/scripts/freesound_download_resample <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/scripts/freesound_download_resample>`_ and follow below steps to download and convert freedsound data into a format expected by the `nemo_asr` collection.

1. We will need some requirements including freesound, requests, requests_oauthlib, joblib, librosa and sox. If they are not installed, please run `pip install -r freesound_requirements.txt`
2. Create an API key for freesound.org at https://freesound.org/help/developers/
3. Create a python file called `freesound_private_apikey.py` and add lined `api_key = <your Freesound api key> and client_id = <your Freesound client id>`
4. Authorize by run `python freesound_download.py --authorize` and visit website and paste response code
5. Feel free to change any arguments in `download_resample_freesound.sh` such as max_samples and max_filesize
6. Run `bash download_resample_freesound.sh <numbers of files you want> <download data directory> <resampled data directory>` . For example: 

.. code-block:: bash

    bash download_resample_freesound.sh 4000 ./freesound ./freesound_resampled_background

Note that downloading this dataset may takes hours. Change categories in download_resample_freesound.sh to include other (speech) categories audio files.
Then, you should have 16khz mono wav files in `<resampled data directory>`. 

Each line is a training example. `audio_filepath` contains path to the wav file, `duration` it's duration in seconds and `text` it's transcript:

.. _google-speechcommand-dataset:

Google Speech Commands Dataset
------------------------------

Google released two versions of the dataset with the first version containing 65k samples over 30 classes and the second containing 110k samples over 35 classes.
We refer to these datasets as v1 and v2

Run the script `process_speech_commands_data.py` to process Google Speech Commands dataset in order to generate files in the supported format of  `nemo_asr`,
which can be found in the `NeMo/scripts/dataset_processing <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/scripts/dataset_processing>`_. 
You should set the data folder of Speech Commands using :code:`--data_root` and the version of the dataset using :code:`--data_version` as an int.

You can further rebalance the train set by passing the `--rebalance` flag.

.. code-block:: bash

    python process_speech_commands_data.py --data_root=<data directory> --data_version=<1 or 2> {--rebalance}


Then, you should have `train_manifest.json`, `validation_manifest.json` and `test_manifest.json`
in the directory `{data_root}/google_speech_recognition_v{1/2}`.

.. note::
  You should have at least 4GB or 6GB of disk space available if you use v1 or v2 respectively. 
  Also, it will take some time to download and process, so go grab a coffee.

Each line is a training example.

.. code-block:: bash

  {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 1.0, "label": "two"}
  {"audio_filepath": "<absolute path to dataset>/two/ec5ab5d5_nohash_2.wav", "duration": 1.0, "label": "two"}



SCF dataset for VAD
--------------------------------
Speech Command & Freesound (SCF) dataset is used to train MarbleNet in paper :cite:`sc-models-marbleboxnet`. Here we show how to download and process it. 
This script assume that you already have the Freesound dataset, if not, have a look at :ref:`_freesound-dataset` . 
We will use the open source  :ref:`_google-speechcommand-dataset` (we will use V2 of the dataset for SCF dataset, but require very minor changes to support V1 dataset) as our speech data. 

These scripts below will download the Google Speech Commands v2 dataset and convert speech and background data to a format suitable for use with nemo_asr.

.. note::
  You may additionally pass :code:`--test_size` or :code:`--val_size` flag for splitting train val and test data. 

  You may additionally pass :code:`--seg_len` flag for indicating the segment length. Default is 0.63s.

  You may additionally pass a :code:`-rebalance_method='fixed|over|under'` at the end of the script to rebalance the class samples in the manifest. 
    


* `'fixed'`: Fixed number of sample for each class. Train 5000, val 1000, and test 1000. (Change number in script if you want)
* `'over'`: Oversampling rebalance method
* `'under'`: Undersampling rebalance method


.. code-block:: bash

    mkdir './google_dataset_v2'
    python process_vad_data.py --out_dir='./manifest/' --speech_data_root='./google_dataset_v2'--background_data_root=<resampled freesound data directory> --log --rebalance_method='fixed' 


After download and conversion, your `manifest` folder should contain a few json manifest files:

* `(balanced_)background_testing_manifest.json`
* `(balanced_)background_training_manifest.json`
* `(balanced_)background_validation_manifest.json`
* `(balanced_)speech_testing_manifest.json`
* `(balanced_)speech_training_manifest.json`
* `(balanced_)speech_validation_manifest.json`

Each line is a training example. `audio_filepath` contains path to the wav file, `duration` it's duration in seconds, `offset` it's offset in seconds, and `label` it's label:

.. code-block:: bash

    {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 0.63, "label": "speech", "offset": 0.0}
    {"audio_filepath": "<absolute path to dataset>/Emergency_vehicle/id_58368 simambulance.wav", "duration": 0.63, "label": "background", "offset": 4.0}



Preparing Custom ASR Data
-------------------------

Preparing Custom Speech Classification Data is almost identical to :ref:`_preprare-custom-asr-data`.

Instead of :code:`text` entry in manifest, you need :code:`label` to determine class of this sample