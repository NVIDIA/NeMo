Datasets
========

.. _ljspeech:

LJSpeech
--------

`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`__ is a speech dataset that
consists of a single female, English speaker. It contains approximately 24
hours of speech.

Obtaining and prepocessing the data for NeMo can be done with
`our helper script <https://github.com/NVIDIA/NeMo/blob/master/scripts/get_ljspeech_data.py>`_:

.. code-block:: bash

    python scripts/get_ljspeech_data.py --data_root=<where_you_want_to_save_data>

.. _csmsc:

Chinese Standard Mandarin Speech Copus(10000 Sentences)
-------------------------------------------------------

`Chinese Standard Mandarin Speech Copus(10000 Sentences) <https://www.data-baker.com/open_source.html>`__ is a speech dataset that
consists of a single female, Mandarin speaker. It contains approximately 12 hours of speech. The copyright of this dataset is held 
by Databaker (Beijing) Technology Co.,Ltd. Support non-commercial use only.

Obtaining and prepocessing the data for NeMo can be done with
`helper script <https://github.com/NVIDIA/NeMo/blob/master/scripts/get_databaker_data.py>`_:

.. code-block:: bash

    python scripts/get_databaker_data.py --data_root=<where_you_want_to_save_data>
