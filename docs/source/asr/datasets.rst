Datasets
========

NeMo has scripts to convert several common ASR datasets into the format expected by the `nemo_asr` collection.
You can get started with those datasets by following the instructions to run those scripts in the section appropriate to each dataset below.

If you have your own data and want to preprocess it to use with NeMo ASR models, check out the `Preparing Custom ASR Data`_ section.

If you already have a dataset that you want to convert to a tarred format, please read the `Tarred Datasets`_ section.

.. _LibriSpeech_dataset:

LibriSpeech
-----------

Run these scripts to download LibriSpeech data and convert it into format expected by `nemo_asr`.
You should have at least 250GB free space.

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_librispeech_data.py --data_root=data --data_set=ALL

After this, your `data` folder should contain wav files and `.json` manifests for NeMo ASR datalayer:


Each line is a training example. `audio_filepath` contains path to the wav file, `duration` it's duration in seconds and `text` it's transcript:

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}

Fisher English Training Speech
------------------------------

Run these scripts to convert the Fisher English Training Speech data into a format expected by the `nemo_asr` collection.

In brief, the following scripts convert the .sph files to .wav, slice those files into smaller audio samples, match the smaller slices with their corresponding transcripts, and split the resulting audio segments into train, validation, and test sets (with one manifest each).

.. note::
  You will need at least 106GB of space to run the .wav conversion, and an additional 105GB for the slicing and matching.
  You will need to have sph2pipe installed in order to run the .wav conversion. 


**Instructions**

These scripts assume that you already have the Fisher dataset from the Linguistic Data Consortium, with a directory structure that looks something like this:

.. code-block:: bash

  FisherEnglishTrainingSpeech/
  ├── LDC2004S13-Part1
  │   ├── fe_03_p1_transcripts
  │   ├── fisher_eng_tr_sp_d1
  │   ├── fisher_eng_tr_sp_d2
  │   ├── fisher_eng_tr_sp_d3
  │   └── ...
  └── LDC2005S13-Part2
      ├── fe_03_p2_transcripts
      ├── fe_03_p2_sph1
      ├── fe_03_p2_sph2
      ├── fe_03_p2_sph3
      └── ...

The transcripts that will be used are located in `fe_03_p<1,2>_transcripts/data/trans`, and the audio files (.sph) are located in the remaining directories in an `audio` subdirectory.

First, convert the audio files from .sph to .wav by running:

.. code-block:: bash

  cd <nemo_root>/scripts
  python fisher_audio_to_wav.py \
    --data_root=<fisher_root> --dest_root=<conversion_target_dir>

This will place the unsliced .wav files in `<conversion_target_dir>/LDC200[4,5]S13-Part[1,2]/audio-wav/`.
It will take several minutes to run.

Next, process the transcripts and slice the audio data:

.. code-block:: bash

  python process_fisher_data.py \
    --audio_root=<conversion_target_dir> --transcript_root=<fisher_root> \
    --dest_root=<processing_target_dir> \
    --remove_noises

This script will split the full dataset into train, validation, and test sets, and place the audio slices in the corresponding folders in the destination directory.
One manifest will be written out per set, which includes each slice's transcript, duration, and path.

This will likely take around 20 minutes to run.
Once finished, you may delete the 10 minute long .wav files if you wish.

2000 HUB5 English Evaluation Speech
-----------------------------------

Run the following script to convert the HUB5 data into a format expected by the `nemo_asr` collection.

Similarly to the Fisher dataset processing scripts, this script converts the .sph files to .wav, slices the audio files and transcripts into utterances, and combines them into segments of some minimum length (default is 10 seconds).
The resulting segments are all written out to an audio directory, and the corresponding transcripts are written to a manifest JSON file.

.. note::
  You will need 5GB of free space to run this script.
  You will also need to have sph2pipe installed.

This script assumes you already have the 2000 HUB5 dataset from the Linguistic Data Consortium.

Run the following to process the 2000 HUB5 English Evaluation Speech samples:

.. code-block:: bash

  python process_hub5_data.py \
    --data_root=<path_to_HUB5_data> \
    --dest_root=<target_dir>

You may optionally include `--min_slice_duration=<num_seconds>` if you would like to change the minimum audio segment duration.

AN4 Dataset
-----------

This is a small dataset recorded and distributed by Carnegie Mellon University, and consists of recordings of people spelling out addresses, names, etc.
Information about this dataset can be found on the `official CMU site <http://www.speech.cs.cmu.edu/databases/an4/>`_.

Please download and extract the dataset (which is labeled "NIST's Sphere audio (.sph) format (64M)" on the site linked above): http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz.

Running the following script will convert the .sph files to .wav using sox, and build one training and one test manifest.

.. code-block:: bash

  python process_an4_data.py --data_root=<path_to_extracted_data>

Once this script finishes, you should have a `train_manifest.json` and `test_manifest.json` in the `<data_root>/an4/` directory.

Aishell1
--------

Run these scripts to download Aishell1 data and convert it into format expected by `nemo_asr`.

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_aishell_data.py --data_root=data

After this, your `data` folder should contain a `data_aishell` folder which contains wav, transcript folder and related `.json` files and `vocab.txt`.

Aishell2
--------

Run the script to process AIShell-2 dataset in order to generate files in the supported format of  `nemo_asr`. You should set the data folder of AIShell-2 using `--audio_folder` and where to push these files using `--dest_folder`.

.. code-block:: bash

    python process_aishell2_data.py --audio_folder=<data directory> --dest_folder=<destination directory>

Then, you should have `train.json` `dev.json` `test.json` and `vocab.txt` in `dest_folder`. 

Preparing Custom ASR Data
-------------------------

The ``nemo_asr`` collection expects each dataset to consist of a set of utterances in individual audio files plus
a manifest that describes the dataset, with information about one utterance per line (.json).
The audio files can be of any format supported by `Pydub <https://github.com/jiaaro/pydub>`_, though we recommend
WAV files as they are the default and have been most thoroughly tested.

You should have one manifest file per dataset that will be passed in, so if you have separate training and validation
datasets, they should also have separate manifests.
Otherwise, you will be loading validation data with your training data and vice versa.

Each line of the manifest should be in the following format:

.. code::

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}

The :code:`audio_filepath` field should provide an absolute path to the .wav file corresponding to the utterance.
The :code:`text` field should contain the full transcript for the utterance, and the :code:`duration` field should
reflect the duration of the utterance in seconds.

Each entry in the manifest (describing one audio file) should be bordered by '{' and '}' and must
be contained on one line.
The fields that describe the file should be separated by commas, and have the form :code:`"field_name": value`,
as shown above.
There should be no extra lines in the manifest, i.e. there should be exactly as many lines in the manifest as
you have audio files in the dataset.

Since the manifest specifies the path for each utterance, the audio files do not have to be located
in the same directory as the manifest, or even in any specific directory structure.

Once you have a manifest that describes each audio file in your dataset, you can then use the dataset by passing
in the manifest file path in your experiment config file, e.g. as `training_ds.manifest_filepath=<path/to/manifest,json>`.

Tarred Datasets
---------------

If you are running experiments on a cluster with datasets stored on a distributed file system, you will likely
want to avoid constantly reading many small files and would prefer tarring your audio files.
There are tarred versions of some NeMo ASR Dataset classes for this case, such as the ``TarredAudioToCharDataset``
(corresponding to the ``AudioToCharDataset``) and the ``TarredAudioToBPEDataset`` (corresponding to the
``AudioToBPEDataset``).
The tarred audio dataset classes in NeMo use `WebDataset <https://github.com/tmbdev/webdataset>`_.

To use an existing tarred dataset instead of a non-tarred dataset, you will need to set ``is_tarred: true`` in
your experiment config file.
Then, you will need to pass in the paths to all of your audio tarballs in ``tarred_audio_filepaths``, either as a list
of filepaths, e.g. ``['/data/shard1.tar', '/data/shard2.tar']``, or in a single brace-expandable string, e.g.
``'/data/shard_{1..64}.tar'`` or ``'/data/shard__OP_1..64_CL_'`` (recommended, see note below).

.. note::
  For brace expansion, there may be cases where ``{x..y}`` syntax cannot be used due to shell interference.
  This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
  Supported opening braces (equivalent to ``{``) are ``(``, ``[``, ``<`` and the special tag ``_OP_``.
  Supported closing braces (equivalent to ``}``) are ``)``, ``]``, ``>`` and the special tag ``_CL_``.
  For SLURM based tasks, we suggest the use of the special tags for ease of use.

As with non-tarred datasets, the manifest file should be passed in ``manifest_filepath``.
The dataloader will assume that the length of the manifest after filtering is the correct size of the dataset
for reporting training progress.

The ``tarred_shard_strategy`` field of the config file can be set if you have multiple shards and are running
an experiment with multiple workers.
It defaults to ``scatter``, which preallocates a set of shards per worker which do not change during runtime.

For more information about the individual tarred datasets and the parameters you can set, including shuffling options,
see the corresponding class APIs on the `Datasets <./api.html#Datasets>`__ page.

.. warning::
  If using multiple workers the number of shards should be divisible by the world size to ensure an even
  split among workers.
  If it is not divisible, logging will give a warning but training will proceed, but likely hang at the last epoch.
  In addition, if using distributed processing, each shard must have the same number of entries after filtering is
  applied such that each worker ends up with the same number of files.
  We currently do not check for this in any dataloader, but your program may hang if the shards are uneven!

Conversion to Tarred Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can easily convert your existing NeMo-compatible ASR datasets using the
`conversion script here <https://github.com/NVIDIA/NeMo/blob/r1.0.0rc1/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_.

.. code::

  python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio>
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=0

This script will shuffle the entries in the given manifest (if ``--shuffle`` is set, which we recommend), filter
audio files according to ``min_duration`` and ``max_duration``, and tar the remaining audio files to the directory
``--target_dir`` in ``n`` shards, along with separate manifest and metadata files.

The files in the target directory will look like:

.. code::

  target_dir/
  ├── audio_1.tar
  ├── audio_2.tar
  ├── ...
  ├── metadata.yaml
  └── tarred_audio_manifest.json

Note that file structures will be flattened such that all audio files are at the top level in each tarball, so to
ensure that filenames are unique in the tarred dataset and the filepaths do not contain "-sub", forward slashes in each ``audio_filepath`` are simply
converted to underscores.
For example, a manifest entry for ``/data/directory1/file.wav`` would be ``_data_directory1_file.wav`` in the tarred
dataset manifest, and ``/data/directory2/file.wav`` would be converted to ``_data_directory2_file.wav``.
