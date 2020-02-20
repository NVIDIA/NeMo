Tutorial
========


Introduction
------------

The goal of Dialog State Tracking  is to build a representation of the status of the ongoing conversation \
being a sequence of utterances exchanged between dialog participants.

In this tutorial we will focus on a multi-domain dialogue MultiWOZ dataset and show how one can train a TRADE model, \
being one of the recent, state of the art models.

The MultiWOZ Dataset
--------------------

The Multi-Domain Wizard-of-Oz dataset (`MultiWOZ`_) is a collection of human-to-human conversations spanning over \
7 distinct domains and containing over 10,000 dialogues.


The TRADE model
---------------


Data Preprocessing
----------------------

First, we need to download the `MULTIWOZ2.1.zip`_ file from the `MultiWOZ`_ project website.


.. _MultiWOZ: https://www.repository.cam.ac.uk/handle/1810/294507

.. _MULTIWOZ2.1.zip: https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y


Next, we need to preprocess and reformat the dataset, what will result in division of data into three splits:

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



Building the NeMo Graph
-----------------------


description of the graph and role of modules


Training and Results
--------------------

description how to train the model and what accuracies one might expect.
