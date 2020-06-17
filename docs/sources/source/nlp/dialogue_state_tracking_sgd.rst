.. _sgd_tutorial:

SGD Tutorial
============

Introduction
------------

A task-oriented dialogue system is a conversational system that can perform a conversation with a user and provide task- (or domain-)specific information. For example, it can book a table in a restaurant or buy a train ticket.
One of the main building blocks of a task-oriented dialogue system is a Dialogue State Tracker (DST).
DST should not only understand what the user just said but also remember what was said before.
DST carries the information about what intent the user has in the conversation, for example, find a restaurant or book a plane ticket,
and what slots along with the corresponding values were mentioned in the dialogue.


The Schema-Guided Dialogue Dataset
----------------------------------

In this tutorial, we are using the Schema-Guided Dialogue (SGD) dataset :cite:`nlp-sgd-rastogi2019towards` that contains over 16k multi-domain goal-oriented conversations across 16 domains.
The data represents conversations between a user and a virtual assistant, and it can be used for various dialogue management tasks:
intent prediction, slot filling, dialogue state tracking, policy imitation learning, language generation. 

One part of the dialogues in the dataset spans across only a single domain dialogues, use ``--task_name dtsc8_single_domain`` to use such dialogues. Another part focuses only on dialogues that span across multiple domains during a single conversation,
``--task_name dstc8_multi_domain`` to train and evaluate on the multi-domain task. ``--task_name dstc8_all`` will use all available dialogues for training and evaluation.

An example of the data format could be found `here <https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/train/dialogues_001.json>`_.
Every dialogue contains the following information:

* **dialogue_id** - a unique dialogue identifier
* **services** - list of services mentioned in the dialogue
* **turns** - a dialogue is comprised of multiple dialogue turns, where a single turn consists of user and systems utterances frames.
* **frames** - each frame contains system or user utterance with assotiated annotraion.
    
    * Each **user** frame containts the following information (values in brackets are from the user frame example in Fig. 1, note some values in the state are coming from the previous dialogue turns):
        
        * **actions** - a list with the following values:
            
            * act - user's intent or act (INFORM)
            * slot - slot names (price_range)
            * values - a list of slot values (moderate)
            * canonical_values (optional) - slot values in their canonicalized form as used by the service
        
        * **service** - service name for the current user utterance (Restaurants_1)
        * **slots** - a list of slot spans in the user utterance, only provided for non-categorical slots. Each slot span contains the following fields:
            
            * slot - non-categorical slot name (city)
            * start/exclusive_end - start/end character index of the non-categorical slot value in the current user utterance (113/122)
        
        * **state** - dialogue state:
            
            * active_intent -  name of an active user intent (FindRestaurants)
            * requested_slots - a list of slots requested be the user in the current turn
            * slot_values - dictionary of slot name - slot value pairs ({"city": ["Palo Alto"], "cuisine": ["American"], "price_range": ["moderate"]}) 
    
    * Each **system** frame containts the following information ((values in brackets are from the system frame example in Fig. 2):
        
        * **actions** - a list with the following values:
            
            * act - system act (OFFER)
            * slot - slot names (restaurant_name)
            * values - a list of slot values (Bird Dog)
            * canonical_values (optional) - slot values in their canonicalized form as used by the service
        
        * **service** - service name for the current turn (Restaurants_1)
        * **service_call** (optional) - request sent to the service:
            
            * method - a name of the intent or function of the service or API being executed (FindRestaurants)
            * parameters - a dictionary of slot name -slot value pairs in their canonicalized form ({"city": ["Palo Alto"], "cuisine": ["American"], "price_range": ["moderate"]})
        
        * **service_results** - results of a service call:
            
            {"city": "Palo Alto",
            "cuisine": "American",
            "has_live_music": "False",
            "phone_number": "650-688-2614",
            "price_range": "moderate",
            "restaurant_name": "Bazille",
            "serves_alcohol": "True",
            "street_address": "550 Stanford Shopping Center"}
        
        * **slots** - a list of slot spans in the system utterance, only provided for non-categorical slots. Each slot span contains the following fields:
            
            * slot - non-categorical slot name (city)
            * start/exclusive_end - start/end character index of the non-categorical slot value in the current user utterance (113/122)

* **speaker** - identifies whether a user or a system is speaking
* **utterance** - user or system utterance

.. figure:: dst_sgd_user_frame.png
    
    Fig. 1: An example of a user frame (source: `a user frame from one of the dialogues <https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/train/dialogues_001.json>`_).



.. figure:: dst_sgd_system_frame.png

    Fig. 2: An example of a system frame (source: `a system frame from one of the dialogues <https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/train/dialogues_001.json>`_).


To find more details and download the dataset, use `this link <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue>`_.

SGD Baseline Model
------------------

The SGD dataset for every dataset split (train, dev, test) provides detailed schema files (see `this for an example here <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/master/train/schema.json>`_).
These files contain information about slots supported by every service, possible values for categorical slots, along with the supported intents.
Besides that, the schemas provide a natural language description of the slots, intents, and services; these descriptions are
utilized by the model to get schema embeddings. Thus, before starting the model training, the training script will create schema embeddings. By default the schema embedding generation
will be performed every time you run the training script, to skip the schema generation step for all subsequent training script runs, use ``--no_overwrite_schema_emb_files``.
(see `nlp/data/datasets/sgd_dataset/schema_processor.py <https://github.com/NVIDIA/NeMo/blob/master/nemo/collections/nlp/data/datasets/sgd_dataset/schema_processor.py>`_ for more implementation details). 

.. figure:: dst_sgd_schema_example.png

    Fig. 3: A schema example for a digital wallet service, (source: :cite:`nlp-sgd-rastogi2019towards`)

Another preprocessing step that could be done once and skipped for all future training runs (if you're not changing anything that could affect it) is the dialogues preprocessing step, i.e. breaking dialogues into dialogue turns and collecting labels and features for a particular turn. Use ``no_overwrite_dial_files``
to overwrite the generated dialogues to skip this step (see `nemo/collections/nlp/data/datasets/sgd_dataset/data_processor.py <https://github.com/NVIDIA/NeMo/blob/master/nemo/collections/nlp/data/datasets/sgd_dataset/data_processor.py>`_ for implementation details).

During training, the Baseline model introduced in :cite:`nlp-sgd-rastogi2019towards` relies on the current user and system utterances and service schemas, compared to the TRADE model that uses all dialogue history.
The SGD model is learning to understand and extract from the dialogue the following things:

- active intent
- requested slots
- categorical slots
- non-categorical slots

Note that for every above mentioned slot, the model predicts slot status and slot value. Only if the slot status is predicted to be active, the associated slot value is taken into account.

Model components:

- **SGDEncoder** - uses a BERT model to encode user utterance. By default, the SGD model uses the pre-trained BERT base cased model from `Hugging Face Transformers <https://huggingface.co/transformers/>`_ to get embedded representations for schema elements and also to encode user utterance. The SGDEncoder returns encoding of the whole user utterance using 'CLS' token and embedded representation of every token in the utterance.
- **SGDDecoder** - returns logits for predicted elements by conditioning on the encoded utterance

FastSGT (Improved Model)
------------------------
We proposed an improved version of the SGD baseline model called Fast Schema Guided Tracker (FastSGT) which is designed and optimzied for seen services.
It has a significantly higher performance in terms of accuracy comparing to the baseline for seen services. FastSGT has the following features:

- Data augmentation for non-categorical slots
- Multi-head attention projection layers for decoders
- In-service slot carry-over mechanism
- Cross-service slot carry-over mechanism
- Ability to make schema embeddings trainable during the model training

Data Augmentation
-----------------
The data augmentation is done offline with `examples/nlp/dialogue_state_tracking/data/sgd/dialogue_augmentation.py <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/dialogue_state_tracking/data/sgd/dialogue_augmentation.py>`_. We used 10x as augmentation factor. It supports modifications on dialogue utterance segments, that are either non-categorical slot values or regular words. When a segment is modified, all future references of the old word in the dialogue are also
altered along with all affected dialogue meta information, e.g. dialogue states, to preserve semantic consistency. This is done by first building a tree structure over the dialogue which stores all relevant meta information.
Currently, we provide one function each for changing either a non-categorical slot value or a regular word:
``get_new_noncat_value()`` is used to replace a non-categorical value by a different value from the same service slot.
``num2str()`` is used to replace a regular word that is a number with its string representation, e.g. '11' becomes 'eleven'.
The script allows the user to easily extend the set of functions by custom ones, e.g. deleting words could be realized by a function that
replaces a regular word by the empty string ''.
The input arguments include configuration settings that determine how many augmentation sweeps are done on the dataset and the probability of modifying a word.
For our experiments we used 9 augmentation sweeps (and concatenated it with the original dataset) at 100% modification rate, resulting in a dataset 10x as large:

.. code-block:: bash

    cd examples/nlp/dialogue_state_tracking/data/sgd
    python dialogue_augmentation.py \
        --input_dir <sgd/train> \
        --repeat 9 \
        --replace_turn_prob 1.0 \
        --replace_word_prob 1.0 \
        --concat_orig_dialogue


Slot Carry-over Mechanisms
--------------------------
The slot carry-over procedures enable the model to retrieve a value for a slot from the preceding system utterance or even previous turns in the dialogue.
There are many cases where the user is accepting some values offered by the system and the value is not mentioned explicitly in the user utterance.
In our system we have implemented two different carry-over procedures.
The value may be offered in the last system utterance, or even in the previous turns. The procedure to retrieve values in these cases is called in-service carry-over.
There are also cases where a switch is happening between two services in multi-domain dialogues.
A dialogue may contain more than one service and the user may switch between these services.
When a switch happens, we may need to carry some values from a slot in the previous service to another slot in the current service.
The carry-over procedure to carry values between two services is called cross-service carry-over.
To support carry-over, we added an status of "carryover" to all the slots which is active when the value of the slot in updated in a turn but it is not explicly mentioned in the current user utterance.
The value for such slots may come from the previous system utterances and offers. We also added an extra value ("#CARRYOVER#") to all the categorical slots.
When a categorical slot has status of "carryover", the value of "#CARRYOVER#" should be predictd for that slot.

    * **In-service carry-over**: We trigger this procedure in three cases: 1-status of a slot is predicted as "carry\_over", 2-the spanning region found for the non-categorical slots is not in the span of the user utterance, 3-"#CARRYOVER#" value is predicted for a categorical slot with "active" or "carry\_over" statuses. The in-service carry-over procedure tries to retrieve a value for a slot from the previous system utterances in the dialogue. We first search the system actions starting from the most recent system utterance and then move backwards for a value mentioned for that slot. The most recent value would be considered for the slot if multiple values are found. If no value could be found, that slot would not get updated in the current state.
    * **Cross-service carry-over mechanism**: Carrying values from previous services to the current one when a switch happens in a turn is done by cross-service carry-over procedure. The previous service and slots are called sources, and the new service and slots are called the targets. To perform the carry-over, we need to build a list of candidates for each slot which contains the slots where a carry-over can happen from them. We create this carry-over candidate list from the training data. We process the whole dialogues in the training data, and count the number of times a value from a source slot and service carry-overs to a target service and slot when a switch happens. We look for the values updated in each turn and check if that value is proposed by the system in the preceding turns. These counts are normalized to the number of the switches between each two services in the whole training dialogues. This carry-over relation between two slots is considered symmetric and statistics from both sides are aggregated. This candidate list for each slot contains a list of slot candidates from other services which is looked up to find a carry-over value. We normalize the number of carry-overs by the number of switches to have a better estimate of the likelihood of carry-overs. In our experiments, the ones with likelihoods less than 0.1 are ignored.

When the carry-over procedures are triggered in a turn, we search for the candidates of each slot to find if any value is mentioned for the slots. If multiple values for a slot are found, the most recent one is used.
The need and effectivness of the carry-over mechanisms are shown by some researches :cite:`nlp-sgd-limiao2019dstc8` and :cite:`nlp-sgd-ruan2020fine`. It improved the accuracy of the state tracker for SGD significantly.
It should be noted that the cross-service carry-over feature does not work for multi-domain dialogues which contain unseen services as
the candidate list is extracted from the training dialogues which does not contain unseen services.
To make it work for unseen services, such transfers can get learned by a model based on the descriptions of the slots :cite:`nlp-sgd-limiao2019dstc8`.

The slot Carry-over mechanisms can be enabled by passing "--state_tracker=nemotracker --add_carry_value --add_carry_status" to the example script.


Training
--------
In order to train the Baseline SGD model on a single domain task and evaluate on its dev and test data, run:

.. code-block:: bash

    cd examples/nlp/dialogue_state_tracking
    python dialogue_state_tracking_sgd.py \
        --task_name dstc8_single_domain \
        --data_dir PATH_TO/dstc8-schema-guided-dialogue \
        --schema_embedding_dir PATH_TO/dstc8-schema-guided-dialogue/embeddings/ \
        --dialogues_example_dir PATH_TO/dstc8-schema-guided-dialogue/dialogue_example_dir \
        --eval_dataset dev_test


Metrics
-------
Metrics used for automatic evaluation of the model :cite:`nlp-sgd-rastogi2020schema`:

- **Active Intent Accuracy** - the fraction of user turns for which the active intent has been correctly predicted.
- **Requested Slot F1** - the macro-averaged F1 score for requested slots over all eligible turns. Turns with no requested slots in ground truth and predictions are skipped.
- **Average Goal Accuracy** For each turn, we predict a single value for each slot present in the dialogue state. This is the average accuracy of predicting the value of a slot correctly.
- **Joint Goal Accuracy** - the average accuracy of predicting all slot assignments for a given service in a turn correctly.

The evaluation results are shown for Seen Services (all services seen during model training), Unseen Services (services not seen during training), and All Services (the combination of Seen and Unseen Services).
Note, during the evaluation, the model first generates predictions and writes them to a file in the same format as the original dialogue files, and then uses these files to compare the predicted dialogue state to the ground truth.

There were some issues in the original evaluation process of the SGD baseline which we fixed.
First, some services were considered seen services during evaluation for single domain dialogues while they do not actually exist in the training data.
The other issue was that the turns which come after an unseen service in multi-domain dialogues could be counted as seen by the original evaluation,
which means errors from unseen services may propagate through the dialogue and affect some of the metrics for seen services.
We fixed it by just considering only turns as by seen services if there are no turns before them in the dialogue by unseen services.
These fixes helped to improve the results. To have a fair comparison we also reported the performance of the baseline model and ours with and without these fixes in the table.


Results on Single Domain
------------------------
The following table shows results of the SGD baseline and that of some NeMo model features. The focus was to improve seen services.
We use * to denote the issue fixed in NeMo that occurred in the original TensorFlow implementation of SGD for single domain.
In the original version of the single domain task, the evaluation falsely classified two services ``Travel_1`` and ``Weather_1`` as Seen Services
although they are never seen in the training data. By fixing this, the Joint Goal Accuracy on Seen Services increased.



Seen Services

+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
|                                                                    |                        Dev set                           |
+                                                                    +-----------------+---------------+-----------+------------+
| SGD baseline implementations                                       | Active Int Acc  | Req Slot F1   | Aver GA   | Joint GA   |
+====================================================================+=================+===============+===========+============+
| Original SGD baseline codebase                                     |      99.06      |     98.67     |   88.08   |    68.58   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo's Implementation of the Baseline                              |      98.91      |     99.60     |   90.71   |    70.94   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker                                       |      98.94      |     99.52     |   95.72   |    85.34   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head                      |      98.99      |     99.66     |   96.26   |    86.81   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + data augmentation                   |      98.89      |     99.70     |   96.23   |    86.53   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head + data augmentation  |     98.95       |     99.70     |   94.96   |    88.06   |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+



Unseen Services

+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
|                                                                    |                        Dev set                           |
+                                                                    +-----------------+---------------+-----------+------------+
| SGD baseline implementations                                       | Active Int Acc  | Req Slot F1   | Aver GA   | Joint GA   |
+====================================================================+=================+===============+===========+============+
| Original SGD baseline codebase                                     |       94.8      |      93.6     |   66.03   |   28.05    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo's Implementation of the Baseline                              |       94.75     |      93.46    |   65.33   |   32.18    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker                                       |      94.74      |    93.49      |   67.55   |   34.68    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head                      |      92.39      |    94.04      |   68.47   |   33.41    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + data augmentation                   |      94.94      |    93.97      |   65.73   |   30.89    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head + data augmentation  |      92.68      |    94.55      |   69.59   |   32.76    |
+--------------------------------------------------------------------+-----------------+---------------+-----------+------------+



All Services

+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
|                                                                   |                        Dev set                           |
+                                                                   +-----------------+---------------+-----------+------------+
| SGD baseline implementations                                      | Active Int Acc  | Req Slot F1   | Aver GA   | Joint GA   |
+===================================================================+=================+===============+===========+============+
| Original SGD trained on single domain task                        |       96.6      |     96.5      |   77.6    |    48.6    |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo's Implementation of the Baseline                             |       96.56     |     96.13     |   76.49   |    49.05   |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker                                      |      96.57      |     96.12     |   79.93   |    56.73   |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head                     |      95.26      |    96.49      |   80.68   |    56.65   |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + data augmentation                  |      96.66      |    96.46      |   79.14   |    55.11   |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+
| NeMo baseline + NeMo Tracker + attention head + data augmentation |      95.41      |    96.79      |   81.47   |    56.83   |
+-------------------------------------------------------------------+-----------------+---------------+-----------+------------+




.. note::
    This tutorial is based on the code from `examples/nlp/dialogue_state_tracking/dialogue_state_tracking_sgd.py  <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/dialogue_state_tracking/dialogue_state_tracking_sgd.py>`_


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-SGD
    :keyprefix: nlp-sgd-
