Datasets
========

HI-MIA
--------

Run the script to download and process hi-mia dataset in order to generate files in the supported format of  `nemo_asr`. You should set the data folder of 
hi-mia using `--data_root`. These scripts are present in <nemo_root>/scripts

.. code-block:: bash

    python get_hi-mia_data.py --data_root=<data directory> 

After download and conversion, your `data` folder should contain directories with follwing set of files as:

* `data/<set>/train.json`
* `data/<set>/dev.json` 
* `data/<set>/{set}_all.json` 
* `data/<set>/utt2spk`