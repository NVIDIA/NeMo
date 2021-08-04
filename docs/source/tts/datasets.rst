TTS Datasets
============

Most TTS scripts work out-of-the-box with the LJ Speech dataset.
However, some models may require supplementary data for training and validation.
The following sections contain details of required data formats and instructions for running preprocessing scripts for
such models.


FastSpeech 2
------------

The FastSpeech 2 model converts from phonemes to Mel Spectrograms and predicts phoneme durations, pitches, and
energies.
Therefore, in addition to the manifest and audio samples, it requires some supplementary files and data:

* A **mappings file** for converting from words to phonemes and phonemes to indices
* **Phoneme durations** (for each audio sample)
* **Pitch per frame** (for each training audio sample)
* **Energy per frame** (for each training audio sample)
* (Optional) **Ignore list** for filtering out samples with OOV words

The ``FastSpeech2Dataset`` uses the manifest format shared with all other NeMo speech tasks.
Each line of the manifest should describe one sample, and should be in the following format:

.. code::

  {"audio_filepath": "/data/wavs/audio.wav", "text": "an unused transcription", "duration": 23.147}

See the documentation on :ref:`Preparing Custom ASR Data` for more details on the NeMo speech manifest format.

.. note::
  The ``FastSpeech2Dataset`` ignores the ``"text"`` field of the manifest, since the model reads phoneme indices from
  the supplementary duration files instead.

The **mappings file** should be a JSON-formatted file that contains dictionaries ``word2phones`` and ``phone2idx``.
``word2phones`` is a mapping that converts from your vocabulary to phonemes.
For example, one entry in ``word2phones`` might be ``"nemo": ["N", "EH1", "M", "OW0"]``.
``phone2idx`` is an (arbitrary) mapping that converts each phoneme to an index for prediction.
It is also used to determine the ``pad_id``, which is set to the length of ``phone2idx``.

Together, these two dictionaries are also used for inference, as an input sentence is first normalized, then converted
to phonemes using ``word2phones``, and then converted from phonemes to indices using ``phone2idx``.

For each audio sample, there should be a corresponding **phoneme durations** file that contains ``token_duration``
(duration per phoneme in frames) and ``text_encoded`` (index of each phoneme corresponding to ``phone2idx``).

For each sample, there should also be a **pitch file** that contains the pitches (F0) per frame, and an **energy file**
that contains the energies (L2-norm of STFT frame amplitudes) per frame.

The **ignore list** is a pickled list of file base names (with no extension) that tells the ``FastSpeech2Dataset``
which audio samples to discard.
You should use this to exclude samples from your manifest that contain OOV words.


Directory Structure
^^^^^^^^^^^^^^^^^^^

While the mappings file and ignore list paths are passed in to the ``FastSpeech2Dataset`` directly, the
``FastSpeech2Dataset`` infers the path to the supplementary files corresponding to each audio sample based on the
audio samples' paths found in the manifest.
For each sample, the paths to the corresponding duration, pitch, and energy files are inferred by replacing ``wavs/``
with ``phoneme_durations/``, ``pitches/``, and ``energies``, and swapping out the file extension (``.wav``) with
``.pt``, ``.npy``, and ``.npy`` respectively.

For example, given manifest audio path ``/data/LJSpeech/wavs/LJ001-0001.wav``, the inferred duration and phonemes file
path would be ``/data/LJSpeech/phoneme_durations/LJ001-0001.pt``.

Your directory structure should therefore look something like this:

.. code::

  data/
  ├── manifest.json
  ├── mappings.json
  ├── ignore_file.pkl
  ├── energies/
  │   ├── basename_1.npy
  │   ├── basename_2.npy
  │   ├── ...
  │   └── basename_n.npy
  ├── phoneme_durations/
  │   ├── basename_1.pt
  │   ├── basename_2.pt
  │   ├── ...
  │   └── basename_n.pt
  ├── pitches/
  │   ├── basename_1.npy
  │   ├── basename_2.npy
  │   ├── ...
  │   └── basename_n.npy
  └── wavs/
      ├── basename_1.wav
      ├── basename_2.wav
      ├── ...
      └── basename_n.wav


LJ Speech Dataset
^^^^^^^^^^^^^^^^^

NeMo comes with scripts for preprocessing the LJ Speech dataset for use with FastSpeech 2, which can be found in the
``<nemo_base>/scripts/dataset_processing/ljspeech/`` directory.
These scripts assume that you have downloaded the LJ Speech dataset and extracted it to ``<ljspeech_base>``.

They perform the following dataset preparation steps:

* Create manifest files and ``.txt`` files with normalized transcripts
* Extract pitches
* Extract energies
* Downloads the `CMU Pronouncing Dictionary <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>`_ to convert normalized
  transcripts to phonemes
* Uses the `Montreal Forced Aligner <https://montreal-forced-aligner.readthedocs.io/en/latest/index.html>`_ (MFA) to
  perform alignment
* Calculates durations based on the alignment
* Generates the mappings file and ignore file for the dataset

To run the scripts, follow the steps below.

#. Download and extract the `LJ Speech dataset <https://keithito.com/LJ-Speech-Dataset/>`_ to ``<ljspeech_base>``.

#. Create the manifest files and normalized text files (for MFA to discover later) by running:

    .. code-block:: bash

      python create_manifests_and_textfiles.py --ljspeech_base=<ljspeech_base>

#. Extract pitches and energies from the audio files:

    .. code-block:: bash

      python extract_ljspeech_energy_pitch.py --ljspeech_base=<ljspeech_base>

    This will write the extracted pitches and energies to ``<ljspeech_base>/pitches/`` and
    ``<ljspeech_base>/energies/``.

#. Run the phoneme extraction and duration calculation script.
   This script will set up a Conda environment to download and install MFA and its dependencies, so make sure that you
   have Anaconda or Miniconda before running the script.

   Also note that MFA does sometimes have trouble finding OpenBlas, so you may have to manually install it with the
   command ``sudo apt-get install libopenblas-dev``.

    .. code-block:: bash

      ./extract_ljspeech_phonemes_and_durs.sh <ljspeech_base>

   This script takes additional options ``--skip_env_setup`` if you have already set up the ``aligner`` environment,
   and ``--g2p_dict=<dict_path>`` if you have already downloaded CMU dict or another G2P dictionary you prefer.

   The alignment step will take a while to run, so be prepared to wait upwards of an hour.

   In addition to alignments and durations, this script will create the mappings and ignore files as well.
   It will also generate some intermediate files that you are free to delete, as you only need the files listed above.

When the following steps are finished, your ``<ljspeech_base>`` directory will look like this.
(Starred files and directories indicate files from intermediate steps that are safe to remove, but that you may want
to keep for bookkeeping purposes.)

.. code::

  <ljspeech_base>/
  ├── alignments/ *
  ├── cmudict.dict *
  ├── energies/
  │   ├── LJ001-0001.npy
  │   └── ...
  ├── ljs_audio_text_test_filelist.txt *
  ├── ljs_audio_text_train_filelist.txt *
  ├── ljs_audio_text_val_filelist.txt *
  ├── ljspeech_test.json
  ├── ljspeech_train.json
  ├── ljspeech_val.json
  ├── mappings.json
  ├── metadata.csv
  ├── phoneme_durations/
  │   ├── LJ001-0001.pt
  │   └── ...
  ├── pitches/
  │   ├── LJ001-0001.npy
  │   └── ...
  ├── README
  ├── uncommented_cmudict.dict *
  ├── wavs/
  │   ├── LJ001-0001.wav
  │   └── ...
  └── wavs_to_ignore.pkl
