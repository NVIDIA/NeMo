Tutorial
========


Introduction
------------

This tutorial explains how to build and train a Dialog State Tracking

Focus on the MultiWOZ dataset.

MultiWOZ Dataset
----------------

The Multi-Domain Wizard-of-Oz dataset (MultiWOZ) is a collection of human-to-human conversations spanning over
multiple domains and topics. 
For the purpose of this tutorial you must download the `MULTIWOZ2.1.zip`_ file from the `MultiWOZ`_ project website.


.. _MultiWOZ: https://www.repository.cam.ac.uk/handle/1810/294507

.. _MULTIWOZ2.1.zip: https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y



Next we need to preprocess and reformat the dataset, what will result in division of data into three splits:

 * traininig split (8242 dialogs in the ``train_dials.json`` file)
 * validation split (1000 dialogs in the ``val_dials.json`` file)
 * test split (999 dialogs in the ``test_dials.json`` file)

In order to preprocess the MultiWOZ dataset you can use the provided `script`_.

.. _script: https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/scripts/multiwoz/process_multiwoz.py

.. note::
    By default, the script assumes that you will copy data from the unpacked archive into the \
    ``~/data/state_tracking/MULTIWOZ2.1/MULTIWOZ2.1/`` \
    folder and will store results in the ``~/data/state_tracking/multiwoz2.1`` folder. \
    Both those can be overriden by passing the command line ``source_data_dir`` and ``target_data_dir`` argumnents \
    respectively.



Training
--------
