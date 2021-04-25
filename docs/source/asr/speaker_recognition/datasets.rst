Datasets
========

.. _HI-MIA:

HI-MIA
------

Run the script to download and process the ``hi-mia`` dataset to generate files in the supported format of ``nemo_asr``. You should 
set the data folder of ``hi-mia`` using ``--data_root``. These scripts are present in the ``<nemo_root>/scripts`` directory.

.. code-block:: bash

    python get_hi-mia_data.py --data_root=<data directory> 

After download and conversion, the ``data`` folder contains directories with the following files:

- ``data/<set>/train.json``
- ``data/<set>/dev.json`` 
- ``data/<set>/{set}_all.json`` 
- ``data/<set>/utt2spk``

All-other Datasets
------------------

The following methods can be applied to any dataset to get similar training manifest files.

#. Prepare ``.scp`` files containing absolute paths to all the ``.wav`` files required for each of the ``train``, ``dev``, and ``test`` 
   set. This can be prepared by using the ``find`` bash command as follows:

   .. code-block:: bash 

       !find {data_dir}/{train_dir}  -iname "*.wav" > data/train_all.scp
       !head -n 3 data/train_all.scp

   Since we created the ``.scp`` file for the ``train`` set, we can:

   - use the ``scp_to_manifest.py`` script to convert the file to a manifest file
   - optionally split the files to ``train`` and ``dev`` for evaluating the models while training by using the ``--split`` flag. We wont 
   be needing the ``--split`` option for the ``test`` folder. 

#. Provide the id number, which is the field num separated by ``/`` to be considered as the speaker label. After the download 
   and conversion, your data folder contains the directories with the following manifest files:
    
   - ``data/<path>/train.json``
   - ``data/<path>/dev.json``
   - ``data/<path>/train_all.json``
    
Each line in the manifest file describes a training sample; ``audio_filepath`` contains the path to the ``.wav`` file - ``duration`` is 
duration in seconds - ``label`` is the speaker class label, for example:

.. code-block:: json
    
    {"audio_filepath": "<absolute path to dataset>/audio_file.wav", "duration": 3.9, "label": "speaker_id"}


Tarred Datasets
---------------

Similarly to ASR, you can tar your audio files and use the ASR sataset class ``TarredAudioToSpeechLabelDataset`` (corresponding to the ``AudioToSpeechLabelDataset``) for this case.

If you want to use a tarred dataset, refer to the `ASR Tarred Datasets <../datasets.html#tarred-datasets>`__ section.