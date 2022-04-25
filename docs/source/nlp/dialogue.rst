.. _dialogue:

Dialogue tasks
======================================

This module consists of various tasks that are related to dialogue. 

**Module Design**

We decided to group dialogue tasks into a common module instead of having a module for each because they share many things in common, meaning that there can be more re-use of code. 
This design can also support easier extension of this module, as developers can work on components of their interest while utilizing other components of dialogue pipeline. 
In particular, we wanted to decouple the task-dependent, model-independent components of DataProcessor and InputExample from the model-dependent, task-independent components of Model and Dataset.

.. image:: dialogue_UML.png
  :alt: Dialogue-UML

**Supported Tasks** 

Supported tasks fall into broad categories of intent / domain classification with slot filling, intent classification as well as sequence generation.

For each category of tasks, there exists several Data Processors to convert raw data from various sources into a common format as well as Dialogue Models that approachs the task in various ways.

Currently, the supported task categories are:

+----------------------------------------------------------+----------------------------------+----------------------------------------------------------------------------------+
| **Task Category**                                        | **Tasks**                        |   **Models**                                                                     |                                                                                     
+----------------------------------------------------------+----------------------------------+----------------------------------------------------------------------------------+
| Domain / Intent Classification                           | Schema Guided Dialogue           | Dialogue GPT Classification Model                                                |
+ with slot filling                                        +----------------------------------+----------------------------------------------------------------------------------+
|                                                          | Assistant                        | SGDQA (BERT-Based Schema Guided Dialogue Question Answering model)               | 
+                                                          +----------------------------------+----------------------------------------------------------------------------------+
|                                                          |                                  | Intent Slot Classification Model                                                 |
+----------------------------------------------------------+----------------------------------+----------------------------------------------------------------------------------+
| Intent Classification                                    | Zero Shot Food Ordering          | Dialogue GPT Classification Model                                                |
+                                                          +----------------------------------+----------------------------------------------------------------------------------+
|                                                          | Omniverse Design                 | Dialogue Nearest Neighbour Model                                                 |
+                                                          +----------------------------------+----------------------------------------------------------------------------------+
|                                                          |                                  | Zero Shot Intent Model (Based on MNLI pretraining)                               |
+----------------------------------------------------------+----------------------------------+----------------------------------------------------------------------------------+
| Sequence Generation                                      | Schema Guided Dialogue Generation| Dialogue GPT Generation Model                                                    |
+                                                          +----------------------------------+----------------------------------------------------------------------------------+
|                                                          | MS Marco NLGen                   | Dialogue S2S Generation Model                                                    |
+----------------------------------------------------------+----------------------------------+----------------------------------------------------------------------------------+

**Configuration** 

Example of model configuration file for training the model can be found at: `NeMo/examples/nlp/dialogue/conf/dialogue_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/dialogue/conf/dialogue_config.yaml>`__.

Because the Dialogue module contains a wide variety of models and tasks, there are a large number of configuration parameters to adjust (some of which only applies to some models/some tasks)

In the configuration file, define the parameters of the training and the model, although most of the default values will work well.
For various task-model combination, only a restricted set of config args will apply. Please read the configuration file for comments on which config args you would need for each model and task.

The configuration can be roughly grouped into a few categories:

- Parameters that describe the training process, such as how many gpus to use: **trainer**
- Parameters that describe the model: **model**
- Parameters that describe optimization: **model.optim**
- Parameters that describe the task: **model.dataset**
- Parameters that describe the dataloaders: **model.train_ds**, **model.validation_ds**, **model.test_ds**,
- Parameters that describe the training experiment manager that log training process: **exp_manager**


Arguments that very commonly need to be edited for all models and tasks

- :code:`do_training`: perform training or only testing
- :code:`trainer.devices`: number of GPUs (int) or list of GPUs e.g. [0, 1, 3]
- :code:`model.dataset.task`: Task to work on [sgd, assistant, zero_shot, ms_marco, sgd_generation, design, mellon_qa]
- :code:`model.dataset.data_dir`: the dataset directory
- :code:`model.dataset.dialogues_example_dir`: the directory to store prediction files
- :code:`model.dataset.debug_mode`: whether to run in debug mode with a very small number of samples [True, False]
- :code:`model.language_model.pretrained_model_name`: language model to use, which causes different Dialogue Models to be loaded (see dialogue.py for details) [t5-base, facebook/bart-large, bert-base-uncased, sentence-transformers/all-MiniLM-L6-v2, microsoft/DialoGPT-small]
- :code:`model.library`: library to load language model from [huggingface or megatron]

**Obtaining data**

Task: Schema Guided Dialogue (SGD) / SGD Generation

:code: `git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git`

Task: MS Marco

Please download the files below and unzip them into a common folder (for model.dataset.data_dir)

https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz
https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz

Task: Assistant 

:code: `git clone https://github.com/xliuhw/NLU-Evaluation-Data`

Then convert the dataset into the required format

.. code::

    python examples/nlp/intent_slot_classification/data/import_datasets.py
        --source_data_dir=`source_data_dir` \
        --target_data_dir=`target_data_dir` \
        --dataset_name='assistant'

- :code:`source_data_dir`: the directory location of the your dataset
- :code:`target_data_dir`: the directory location where the converted dataset should be saved


Unfortunately other datasets are currently not available publically 

**Training/Testing a model**

Please try the example Dialogue model in a Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/nlp/dialogue.ipynb>`__).

Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

An example script on how to train the model can be found here: `NeMo/examples/nlp/dialogue/dialogue.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/dialogue/dialogue.py>`__.

The following is an example of the command for training the model:

.. code::

    python examples/nlp/dialogue/dialogue.py \
           do_training=True \
           model.dataset.task=sgd \
           model.dataset.debug_mode=True \
           model.language_model.pretrained_model_name=gpt2 \
           model.data_dir=<PATH_TO_DATA_DIR> \
           model.dataset.dialogues_example_dir=<PATH_TO_DIALOGUES_EXAMPLE_DIR> \
           trainer.devices=[0] \
           trainer.accelerator='gpu'
