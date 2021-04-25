Datasets
========

NeMo includes scripts to convert several common ASR datasets into the format expected by the ``nemo_asr`` collection. You can get 
started with those datasets by following the instructions to run those scripts in the section appropriate to each dataset below.

If you have your own data and want to preprocess it to use with NeMo ASR models, check out the `Preparing Custom Speech Classification Data`_ section at the bottom of the page.

.. _Freesound-dataset:

Freesound
---------

`Freesound <http://www.freesound.org/>`_ is a website that aims to create a huge open collaborative database of audio snippets, 
samples, recordings, and bleeps. Most audio samples are released under Creative Commons licenses that allow their reuse. Researchers 
and developers can access Freesound content using the Freesound API to retrieve meaningful sound information such as metadata, analysis 
files, and the sounds themselves. 

#. Go to ``<NeMo_git_root>/scripts/freesound_download_resample`` and perform the following steps to download and convert the freedsound 
   data into a format expected by the ``nemo_asr`` collection.

#. Install the required libraries including freesound, requests, requests_oauthlib, joblib, librosa and sox.
   ``pip install -r freesound_requirements.txt``.

#. Create an API key for freesound.org at https://freesound.org/help/developers/.

#. Create a Python file called ``freesound_private_apikey.py`` and add lined ``api_key = <your Freesound api key>`` and 
   ``client_id = <your Freesound client id>``.

#. Authorize the freesound download, visit the website, and paste the response code. 
   ``python freesound_download.py --authorize``

#. Change any arguments in ``download_resample_freesound.sh`` such as ``max_samples`` and ``max_filesize``.

#. Run ``bash download_resample_freesound.sh <numbers of files you want> <download data directory> <resampled data directory>``. For 
   example: 

  .. code-block:: bash

      bash download_resample_freesound.sh 4000 ./freesound ./freesound_resampled_background

  Note that downloading this dataset may take hours. Change categories in ``download_resample_freesound.sh`` to include other (speech) 
  category audio files; you should have 16khz mono ``.wav`` files in the ``<resampled data directory>`` directory. 

.. _Google-Speech-Commands-Dataset:

Google Speech Commands Dataset
------------------------------

Google released two versions of the dataset with the first version containing 65k samples over 30 classes and the second containing 
110k samples over 35 classes. We refer to these datasets as ``v1`` and ``v2`` respectively.

#. Run the script ``process_speech_commands_data.py`` to process the Google Speech Commands dataset in order to generate files in the 
   supported format of ``nemo_asr``, which can be found in ``<NeMo_git_root>/scripts/dataset_processing/``. 

#. Set the data folder of Speech Commands using :code:``--data_root`` and the version of the dataset using :code:``--data_version`` as 
   an int.

You can further rebalance the train set by randomly oversampling files inside the manifest by passing the ``--rebalance`` flag.

.. code-block:: bash

    python process_speech_commands_data.py --data_root=<data directory> --data_version=<1 or 2> {--rebalance}


Then, you should have ``train_manifest.json``, ``validation_manifest.json``, and ``test_manifest.json`` in the directory 
``{data_root}/google_speech_recognition_v{1/2}``.

.. note::
  You should have at least 4 GB or 6 GB of disk space available if you use ``v1`` or ``v2`` respectively. Also, it will take some 
  time to download and process.

Each line below is a training example:

.. code-block:: bash

  {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 1.0, "label": "two"}
  {"audio_filepath": "<absolute path to dataset>/two/ec5ab5d5_nohash_2.wav", "duration": 1.0, "label": "two"}


Speech Command and Freesound for VAD
------------------------------------

The Speech Command and Freesound (SCF) dataset is used to train MarbleNet in the `paper <https://arxiv.org/pdf/2010.13886.pdf>`_. Here 
we show how to download and process it. This script assumes that you already have the Freesound dataset, if not, have a look at 
:ref:`Freesound-dataset`. We will use the open-source :ref:`Google-Speech-Commands-Dataset` (we will use V2 of the dataset for SCF 
dataset, but require very minor changes to support the V1 dataset) as our speech data. 

These scripts download the Google Speech Commands v2 dataset and convert the speech and background data to a format suitable for use 
with ``nemo_asr``.

.. note::
  You can additionally pass the :code:``--test_size`` or :code:``--val_size`` flag for splitting train val and test data. 

  You can additionally pass the :code:``--seg_len`` flag for indicating the segment length. Default is ``0.63s``.

  You can additionally pass a :code:``-rebalance_method='fixed|over|under'`` at the end of the script to rebalance the class samples in the manifest. 

- ``fixed``: Fixed number of samples for each class. Train 5000, val 1000, and test 1000. (Change the number in the script if you want.)
- ``over``: Oversampling rebalance method.
- ``under``: Undersampling rebalance method.

.. code-block:: bash

    mkdir './google_dataset_v2'
    python process_vad_data.py --out_dir='./manifest/' --speech_data_root='./google_dataset_v2'--background_data_root=<resampled freesound data directory> --log --rebalance_method='fixed' 

After download and conversion, your ``manifest`` folder should contain a few json manifest files:

- ``(balanced_)background_testing_manifest.json``
- ``(balanced_)background_training_manifest.json``
- ``(balanced_)background_validation_manifest.json``
- ``(balanced_)speech_testing_manifest.json``
- ``(balanced_)speech_training_manifest.json``
- ``(balanced_)speech_validation_manifest.json``

Each line is a training example. ``audio_filepath`` contains a path to the ``.wav`` file, ``duration`` is duration in seconds, ``offset`` 
is offset in seconds, and ``label`` is label (class):

.. code-block:: bash

    {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 0.63, "label": "speech", "offset": 0.0}
    {"audio_filepath": "<absolute path to dataset>/Emergency_vehicle/id_58368 simambulance.wav", "duration": 0.63, "label": "background", "offset": 4.0}


Preparing Custom Speech Classification Data
--------------------------------------------

Preparing Custom Speech Classification Data is almost identical to `Preparing Custom ASR Data <../datasets.html#preparing-custom-asr-data>`__.

Instead of :code:``text`` entry in manifest, you need :code:``label`` to determine the class of this sample.

Tarred Datasets
---------------

Similarly to ASR, you can tar your audio files and use the ASR dataset class ``TarredAudioToClassificationLabelDataset`` 
(corresponding to the ``AudioToClassificationLabelDataset``) for this case.

If you would like to use a tarred dataset, refer to the `ASR Tarred Datasets <../datasets.html#tarred-datasets>`__ section.