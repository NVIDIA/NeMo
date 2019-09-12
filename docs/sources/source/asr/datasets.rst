Datasets
========

.. _LibriSpeech_dataset:

LibriSpeech
-----------

Run these scripts to download LibriSpeech data and convert it into format expected by `nemo_asr`.
You should have at least 110GB free space.

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
The resulting segments are all written out to an audio directory, and the corresponding transcripts are written to a manifest JSON.

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
