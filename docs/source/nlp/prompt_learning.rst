Prompt Tuning
-------------

Prompt tuning is a continuous or soft prompt approach to finding the optimal prompt for a specific prompting-based tasks. Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning utilizes continuous prompt tokens that can be optimized via gradient decent. In addition to increased task performance compared to discrete prompting methods, prompt tuning has been shown to yield performance competitive with finetuning all of a model’s parameters for T5 style models greater than 10B parameters. This is particularly exciting because prompt tuning typically involves tuning parameters amounting to less then 1% of the original model’s size. A model can also be prompt tuned for multiple tasks simultaneously without the risk of over fitting on any one task leading to a degradation in performance on other tasks. With these benefits, prompt tuning can be used as a lighter weight and more flexible alternative to full model finetuning. Prompt tuning can also be used additively with other discrete prompt selection methods.

Implementation Overview
^^^^^^^^^^

Our current prompt tuning implementation adapt’s Lester et. al’s EMNLP 2021 "`The Power of Scale for Parameter-Efficient Prompt Tuning <https://arxiv.org/abs/2104.08691>`_" to prompt tuning for GPT style models. In this implementation, a number of soft tokens specified by the user are prepended to the beginning of the discrete token input embeddings during the forward pass. During training, all model parameters are frozen except for those corresponding to the soft tokens. Only the soft prompt parameters are updated via gradient decent in the backward pass. Each soft token has the same dimensionality as a regular token embedding from the model’s vocabulary corresponding to the ``hidden_size`` hyperparameter. Soft token embeddings can be initialized randomly or with selected existing embeddings from the pretrained model. 

As of NeMo 1.7 prompt tuning now works with tensor parallel > 1. 

Data Formatting
^^^^^^^^^^

The dataset should be a .jsonl file where each json object has 3 fields: ``prompt_tag``, ``text``, and ``answer``.

.. code::

  {"prompt_tag": [tag1], "text": [text1], "answer": [answer1]}
  {"prompt_tag": [tag1], "text": [text2], "answer": [answer2]}
  {"prompt_tag": [tag1], "text": [text3], "answer": [answer3]}
  
.. _data-example-label:

Prompt Tuning Specific Config Values
^^^^^^^^^^
.. list-table:: Prompt Tuning Config Parameters
   :widths: 15 15 25
   :header-rows: 1
   
   * - **Parameter**
     - **Data type**
     - **Description**
   * - **restore_from_path**
     - string
     - Path to a .nemo file for a pretrained GPT model
   * - **model.use_soft_prompts**
     - bool
     - Flag indicating whether to use prompt tags. Must be set to true if doing prompt tuning or if you want to existing prompt tags during inference. 
   * - **model.num_prompt_tokens**
     - int
     - The number of soft prompt tokens that will be initialized and prepended to all model inputs. Must be consistent across all prompt tuning tasks.
   * - **model.new_prompt_tags**
     - list of strings
     - A name associated with the task for which you're currently prompt tuning the model. This is used to prepend the correct soft prompt to a corresponding model input and must match the prompt tag associated with the text inputs for that task. See `Data Formatting`_ for an example. Currently prompt tuning on only one task at a time is                supported, but inference can be performed on multiple tasks at once. 
   * - **model.existing_prompt_tags**
     - list of strings
     - list of existing, already tuned soft prompt tags. Only needs to be set when a model has been prompt tuned on a task previously and you want to tune it on another task.
   * - **model.new_prompt_init_methods**
     - list of strings
     - Either ``['text']`` or ``['random']`` corresponding to initializing soft prompt embeddings from existing token embeddings or randomly. ``['text']`` is recommended. 
   * - **model.new_prompt_init_text**
     - list of strings
     - The text you want to use for soft prompt initalization if ``model.new_prompt_init_methods`` is set to ['text']. The text is tokenized and clipped or tiled to match ``model.num_prompt_tokens``. The vocab embeddings associated with each token are copied and use to initialize the soft prompts.
   * - **model.calc_loss_on_answer_only**
     - bool
     - Whether to calculate cross entropy loss on the full text input or only the answer portion of the input during prompt tuning. 
   * - **model.data.train_ds**
     - string
     - path to training dataset .json or .jsonl file. See `Data Formatting`_ for an example
   * - **model.data.valid_ds**
     - string
     - path to validation dataset .json or .jsonl file. See `Data Formatting`_ for an example
   

Example Prompt Tuning Command for the First Task
^^^^^^^^^^

.. code::
  
  EXPR_NAME='winogrande_prompt_tuning'
  RESTORE_PATH='megatron_gpt.nemo'
  GPUS=1
  MAX_STEPS=1000
  PROMPT_LENGTH=150
  
  echo "Prompt tuning starting"
  python megatron_gpt_prompt_tuning.py \
          --config-name=megatron_gpt_config \
          trainer.devices=$GPUS \
          trainer.accelerator='gpu' \
          trainer.max_steps=$MAX_STEPS \
          restore_from_path=$RESTORE_PATH \
          exp_manager.name=$EXPR_NAME \
          exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
          +model.use_soft_prompts=True \
          +model.num_prompt_tokens=$PROMPT_LENGTH \
          +model.new_prompt_tags=['Winogrande'] \
          +model.new_prompt_init_text=['disambiguate pronoun noun names pick correct name fill blank'] \
          +model.new_prompt_init_methods=['text'] \
          model.data.data_prefix=None \
          +model.data.train_ds='winogrande_prompt_tuning_train.jsonl' \
          +model.data.valid_ds='winogrande_prompt_tuning_val.jsonl' \
          +model.data.batch_size=32 \
          model.optim.lr=2e-3 \
          model.optim.sched.min_lr=2e-6 \
          model.optim.sched.warmup_steps=320 \
          model.optim.sched.constant_steps=2240 \
          model.encoder_seq_length=2048

Example Prompt Tuning Command for the Second Task
^^^^^^^^^^

Be sure to update ``model.existing_prompt_tags`` with tags from previous prompt tuning run
and to use the .nemo file saved at the end of the last prompt tuning run.

.. code::

  EXPR_NAME='rte_prompt_tuning'
  RESTORE_PATH='winogrande_prompt_tuning.nemo'
  GPUS=1
  MAX_STEPS=780
  PROMPT_LENGTH=150
  VAL_CHECK_INTERVAL=50

  echo "Prompt tuning starting"
  python megatron_gpt_prompt_tuning.py \
          --config-name=megatron_gpt_config \
          trainer.devices=$GPUS \
          trainer.accelerator='gpu' \
          trainer.max_steps=$MAX_STEPS \
          trainer.val_check_interval=$VAL_CHECK_INTERVAL \
          restore_from_path=$RESTORE_PATH \
          exp_manager.name=$EXPR_NAME \
          exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
          +model.use_soft_prompts=True \
          +model.num_prompt_tokens=$PROMPT_LENGTH \
          +model.existing_prompt_tags=['Winogrande'] \
          +model.new_prompt_tags=['RTE'] \
          +model.new_prompt_init_text=['entailment cause relationship imply label text'] \
          +model.new_prompt_init_methods=['text'] \
          model.data.data_prefix=None \
          +model.data.train_ds='RTE_prompt_tuning_train.jsonl' \
          +model.data.valid_ds='RTE_prompt_tuning_val.jsonl' \
          +model.data.batch_size=32 \
          model.optim.lr=2e-4 \
          model.optim.sched.min_lr=2e-6 \
          model.optim.sched.warmup_steps=78 \
          model.optim.sched.constant_steps=545 \
          model.encoder_seq_length=2048


Example Prompt Tuned Inference
^^^^^^^^^^
The inference file can contain a mix of prompts from all the tasks the model has been prompt tuned on. 

.. code::

    python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \


Example prompt tuning script: `NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py>`__.

Example prompt tuned inference script: `NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/megatron_gpt_eval.py>`__.
