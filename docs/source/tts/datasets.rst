Data Preprocessing
==================

NeMo TTS recipes support most of public TTS datasets that consist of multiple languages, multiple emotions, and multiple speakers. Current recipes covered English (en-US), German (de-DE), Spanish (es-ES), and Mandarin Chinese (work in progress), while the support for many other languages is under planning. NeMo provides corpus-specific data preprocessing scripts, as shown in the directory of `scripts/data_processing/tts/ <https://github.com/NVIDIA/NeMo/tree/main/scripts/dataset_processing/tts/>`_, to convert common public TTS datasets into the format expected by the dataloaders as defined in `nemo/collections/tts/torch/data.py <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/torch/data.py>`_. Following the instructions on how to run the corpus-specific scripts below, you can get started with either directly processing those public datasets, or creating your custom scripts to preprocess your custom datasets scripts.

Public TTS Datasets
------------------------------
This table below summarizes the statistics for a collection of high-quality public datasets used by NeMo TTS. We recommend to start customizing the scripts for your custom datasets that have close sampling rate and number of speakers.

.. csv-table::
   :file: data/datasets.csv
   :align: right
   :header-rows: 1

Corpus-Specific Data Preprocessing
----------------------------------
NeMo implements model-agnostic data preprocessing scripts that wrap up steps of **downloading raw datasets, extracting files, and/or normalizing raw texts, and generating data manifest files**. Most scripts are able to be reused for any datasets with only minor adaptations. Most TTS models work out-of-the-box with the LJSpeech dataset, so it would be straightforward to start adapting your custom script from `LJSpeech script <https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/ljspeech/get_data.py>`_. For some models that may require supplementary data for training and validating, such as speech/text alignment prior, pitch, speaker ID, emotion ID, energy, etc, you may need an extra step of **supplementary data extraction** by calling `script/dataset_processing/tts/extract_sup_data.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/extract_sup_data.py>`_ . The following sub-sections demonstrate detailed instructions for running data preprocessing scripts.

LJSpeech
~~~~~~~~
* Dataset URL: https://keithito.com/LJ-Speech-Dataset/
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/ljspeech/get_data.py
* Command Line Instruction:

.. code-block:: shell-session

    $ python scripts/dataset_processing/tts/ljspeech/get_data.py \
        --data-root <your_local_dataset_root> \
        --whitelist-path <your_local_whitelist_filepath> \
        or default nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path ljspeech/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>


LibriTTS
~~~~~~~~
* Dataset URL: https://www.openslr.org/60/
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/libritts/get_data.py
* Command Line Instruction:

.. code-block:: console

    $ python scripts/dataset_processing/tts/libritts/get_data.py \
        --data-root <your_local_dataset_root> \
        --data-sets dev_clean

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path ljspeech/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>



HiFiTTS
~~~~~~~
The texts of this dataset has been normalized already. So there is no extra need to preprocess the data again. But we still need a download script and split it into manifests.

* Dataset URL: http://www.openslr.org/109/
* Dataset Processing Script: TBD
* Command Line Instruction: TBD


Thorsten Müller (German Neutral-TTS dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Dataset URL: https://www.openslr.org/resources/95/
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/openslr/get_data.py
* Command Line Instruction:

.. code-block:: bash

    $ python scripts/dataset_processing/tts/openslr/get_data.py \
        --data-root <your_local_dataset_root> \
        --val-size 0.1 \
        --test-size 0.2 \
        --seed-for-ds-split 100

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path openslr/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>

HUI Audio Corpus German
~~~~~~~~~~~~~~~~~~~~~~~
* Dataset URL: https://opendata.iisys.de/datasets.html
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/tts/hui_acg/get_data.py
* Command Line Instruction:

.. code-block:: bash

    $ python scripts/dataset_processing/tts/hui_acg/get_data.py \
        --data-root <your_local_dataset_root> \
        --set-type clean \
        --min-duration 0.1 \
        --max-duration 15 \
        --val-num-utts-per-speaker 1 \
        --test-num-utts-per-speaker 1 \
        --seed-for-ds-split 100

    $ python scripts/dataset_processing/tts/hui_acg/phonemizer.py \
        --json-manifests <your_path_to_train_manifest> <your_path_to_val_manifest> <your_path_to_test_manifest> \
        --preserve-punctuation

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path hui_acg/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>