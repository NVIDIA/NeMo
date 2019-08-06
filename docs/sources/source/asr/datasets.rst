Datasets
========

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

Mozilla Common Voice
--------------------
coming soon ...

WSJ
---
coming soon ...

Switchboard and CallHome
------------------------
coming soon ...

Building Your Own Dataset
-------------------------
coming soon ...


