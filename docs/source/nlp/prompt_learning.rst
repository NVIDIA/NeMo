Prompt Learning
-------------

Within NeMo we refer to **p-tunining** and **prompt tuning** methods collectivly as prompt learning. Both methods are parameter efficient alternatives to fine-tuning pretrained language models. Our NeMo implementation makes it possible to use one pretrained GPT model on many downstream tasks without needing to tune the model's full set of parameters. It also allows for adding new tasks to your model without overwriting or disrupting previous tasks for which the model has already been p-tuned/prompt-tuned. Because the original model parameters are frozen and never altered by either method, p-tuning/prompt-tuning also avoid cartographic forgetting issues often encountered when fine-tuning models.

- Our prompt tuning implementation is based off Lester et. al’s EMNLP 2021 paper "`The Power of Scale for Parameter-Efficient Prompt Tuning <https://arxiv.org/abs/2104.08691>`_"
- Our p-tuning implementation is based off Liu et al's paper "`GPT Understands, Too <https://arxiv.org/abs/2103.10385>`_"

Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning and p-tuning utilize virtual prompt tokens that can be optimized via gradient decent. The only difference between prompt tuning and p-tuning within NeMo-Megatron is the architecture used to tune the soft prompt tokens during training.

Terminology
^^^^^^^^^^
We will be using the terms ``continuous``, ``soft``, and ``virtual`` interchangeably to refer to tokens inserted into the model prompt that have no concrete mapping to strings or characters within the model’s vocabulary. These virtual tokens exist in contrast to the ``discrete``, ``hard``, or ``real`` tokens that do make up the model’s vocabulary. Virtual tokens are purely 1D vectors with dimensionality equal to that of each real token embedding, matching the ``hidden_size`` hyperparameter. In training and inference, continuous token embeddings are inserted amoung discrete token embeddings according to a template you provide in the model's config. We will demonstrate how to do this below.

When referring to p-tuning and prompt tuning together, we will be using the phrase prompt learning for simplicity.

Prompt Tuning
^^^^^^^^^^^^^

When prompt-tuning a pretrained GPT model, soft prompt embeddings are initialized as a 2D matrix of size ``total_virtual_tokens X hidden_size``. Each task the model is prompt-tuned to perform has its own 2D embedding matrix associated with it. Tasks do not share any parameters during traning or inference. All GPT model parameters are frozen and only the embedding parameters for each task are updated during training.

In prompt tuning you can specify how the embeddings are initialized for each task. You can either

- Initialize embedding parameters according to some random distribution
- Initialize embedding parameters from existing vocabulary embeddings (recommended)

If you choose to initialize virtual token embeddings from existing embedding weights, you can provide the string of words you want to use for initialization in the model's config. This string will be tokenized and tiled or truncated to match the specified number of virtual tokens you would like to use (``total_virtual_tokens``). Vocab embeddings are copied and used to initialize the soft prompt embedding matrix for each task. The vocab embeddings themselves are not updated or changed during prompt tuning.

P-Tuning
^^^^^^^^

When p-tuning, an LSTM model is used to predict virtual token embeddings. LSTM parameters are randomly initialized at the start of p-tuning. All GPT model parameters are frozen, and only the LSTM weights are updated at each training step. LSTM parameters are shared between all tasks that are p-tuned at the same time, but the LSTM model outputs unique virtual token embeddings for each task. The virtual tokens predicted by the LSTM are inserted among the discrete token input in the exact same manner as with prompt-tuning. You still specify the number of virtual tokens you want to use by setting ``total_virtual_tokens`` and each virtual token embedding is still a 1D vector of size ``hidden_size``.

Using Both Prompt and P-Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single pretrained GPT model can use both p-tuning and prompt-tuning. While you must decide to use either p-tuning or prompt-tuning for each task you want your model to perform, you can p-tune your model on a set of tasks *A*, then prompt tune your same model on a different set of tasks *B*, then finally run inference on tasks from both *A* and *B* at the same time. During prompt-tuning or p-tuning, tasks tuned at the same time must use the same number of virtual tokens. During inference, tasks using differing amounts of virtual tokens can be run at the same time.

P-tuning usually requires fewer virtual tokens per task to achieve good results but uses a higher number of parameters (~6% original model params) compared to prompt-tuning (~.06% of original model params). Because p-tuning shares parameters between tasks during training, p-tuning on many very different tasks at once might perform worse than prompt tuning, which tunes a distinct set of parameters for each task. For the same reason, p-tuning your model on multiple tasks that are similar might allow your model to share insight between tasks.

When p-tuning completes, prompt tuned virtual tokens from the p-tuning ``prompt_encoder`` are automatically moved to the ``prompt_table`` where all prompt tuned and p-tuned soft prompts are stored. The LSTM ``prompt_encoder`` is then removed from the model. This allows us to preserve previously p-tuned soft prompts while still maintaining the ability to add new p-tuned or prompt-tuned soft prompts in the future. The ``prompt_table`` uses the ``taskname`` as a key to look up the correct virtual tokens for a specified task. The ``prompt_table``'s hash table data structure also makes it possible for each task to flexibly use a different amount of virtual tokens. 

Dataset Preprocessing
^^^^^^^^^^^^^^^^^^^^^

The prompt learning dataset accepts a list of json/dictionary objects or a list of json file names where each json file contains a collection of json objects. Each json object must include the field ``taskname`` which is a string identifier for the task the data example corresponds to. They should also include one or more fields corresponding to different sections of the discrete text prompt. The input data might look like:

.. code::

  [
    {"taskname": "squad", "context": [CONTEXT_PARAGRAPH_TEXT1], "question": [QUESTION_TEXT1], "answer": [ANSWER_TEXT1]},
    {"taskname": "squad", "context": [CONTEXT_PARAGRAPH_TEXT2], "question": [QUESTION_TEXT2], "answer": [ANSWER_TEXT2]},
    {"taskname": "intent_and_slot", "utterance": [UTTERANCE_TEXT1], "label": [INTENT_TEXT1][SLOT_TEXT1]},
    {"taskname": "intent_and_slot", "utterance": [UTTERANCE_TEXT2], "label": [INTENT_TEXT2][SLOT_TEXT2]},
    {"taskname": "sentiment", "sentence": [SENTENCE_TEXT1], "label": [SENTIMENT_LABEL1]},
    {"taskname": "sentiment", "sentence": [SENTENCE_TEXT2], "label": [SENTIMENT_LABEL2]},
  ]
  
These additional fields can be unlimited in number and will be used to help map different parts of the discrete text input to a prompt template that you define. We show how this mapping works and how to construct your prompt template in the `_Prompt_Formatting_` section. Data examples for each dataset can all be passed to the dataset class in one file, or in seprate ``.jsonl`` files in a list. 
  
.. _data-example-label:

Prompt Formatting
^^^^^^^^^^^^^^^^^

To customize different prompts for different tasks, we simply need to specify the prompt task template in the config file at ``model.task_templates``. The virtual token markers ``<|VIRTUAL_PROMPT_#|>`` signify where you want virtual tokens to be placed in the template string. ``<|VIRTUAL_PROMPT_0|>``, ``<|VIRTUAL_PROMPT_1|>``, and ``<|VIRTUAL_PROMPT_2|>`` indicate where a number of virtual tokens matching the values given at ``virtual_token_splits[0]``, ``virtual_token_splits[1]`` and ``virtual_token_splits[2]`` will be placed. The other variable fields ``{var}`` refer to the fields in the data json.

For example, given:

- the data json ``{"sentence1": "And he said, Mama, I'm home.", "sentence2": "He didn't say a word."}``
- virtual token splits set to ``virtual_token_splits = [3, 3, 3]``
- a prompt template set to ``prompt_template = "<|VIRTUAL_PROMPT_0|> Hypothesis: [sentence1], <|VIRTUAL_PROMPT_1|> Premise: [sentence2] <|VIRTUAL_PROMPT_2|> Answer:"``

the input will be translated into ``VVV Hypothesis: And he said, Mama, I'm home. VVV Premise: He didn't say a word. VVV Answer:``, where ``VVV`` are three virtual tokens.

.. code::

  config.model.task_templates = [
    {
        "taskname": "sentiment",
        "prompt_template": "<|VIRTUAL_PROMPT_0|> {sentence} sentiment: {label}",
        "total_virtual_tokens": 10,
        "virtual_token_splits": [10],
        "truncate_field": "sentence"
    },
    {
        "taskname": "intent_and_slot",
        "prompt_template": "<|VIRTUAL_PROMPT_0|> Predict intent and slot <|VIRTUAL_PROMPT_1|> :\n{utterance}{label}",
        "total_virtual_tokens": 10,
        "virtual_token_splits": [7, 3],
        "truncate_field": None
    }
  ]

.. _prompt-formatting-label:

``model.task_templates`` Config Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table:: 
    :widths: 15 15 25
    :header-rows: 1
    
    * - **Parameter**
      - **Data type**
      - **Description**
    * - **taskname**
      - string
      - Short string denoting the task, used to lookup task specific virtual tokens from the ``prompt_table``. Refers to the same ``taskname`` in the dataset json objects.
    * - **prompt_template**
      - string
      - a string showing the model where to place virtual tokens and how to map dataset json fields to where they belong in the model prompt
    * - **total_virtual_tokens**
      - int
      - specifies the total number of virtual tokens that will be inserted into the model prompt
    * - **virtual_token_splits**
      - list of ints
      - specifies the number of virtual tokens that belong at each ``<|VIRTUAL_PROMPT_#|>`` marker. ``virtual_token_splits`` values should add up to ``total_virtual_tokens``. The number of ``virtual_token_splits`` should match the number of ``<|VIRTUAL_PROMPT_#|>`` markers.
    * - **truncate_field** 
      - string
      - specifies which field in the data json to truncate if the length of the input exceeds the maximum sequence length of the model. If ``truncate_field`` is set to ``None``, examples that are too long are simply dropped from the dataset.

Prompt Learning Specific Config Values
^^^^^^^^^^
.. list-table:: Prompt Tuning Config Parameters
   :widths: 15 15 15 25
   :header-rows: 1
   
   * - **Parameter**
     - **Data type**
     - **Default**
     - **Description**
   * - **model.nemo_path**
     - string
     - ``${name}.nemo`` the name of the experiment
     - Path to where you want to save your model after prompt tuning/p-tuning, must end in `.nemo`
   * - **model.lm_finetune**
     - bool
     - False
     - whether fine tune all the GPT language model weights
   * - **model.virtual_prompt_style**
     - string
     - 'p-tuning'
     - one of 'prompt-tuning', 'p-tuning', or 'inference'
   * - **model.language_model_path**
     - string
     - None
     - Path to the GPT language model .nemo file you want to use for prompt learning, not needed if ``restore_path`` is set
   * - **model.restore_path**
     - string
     - None
     - Path to a .nemo file of existing ``MegatronGPTPromptLearningModel`` that has already been prompt tuned or p-tuned on at least one task. P-tuned or prompt tuned in this training session will be added to this model's `prompt_table`.
   * - **model.new_tasks**
     - list of strings
     - ``[]``
     - List of new tasknames to be prompt or p-tuned
   * - **model.existing_tasks**
     - list of strings
     - ``[]``
     - List of tasks the model has already been p-tuned/prompt-tuned for, needed when a restore path is given
   * - **model.task_templates**
     - list
     - required
     - See the ``model.task_templates`` Config Parameters Table above
   * - **model.prompt_tuning.new_prompt_init_methods**
     - list of strings
     - ``[]``
     - List of 'text' or 'random', should correspond to the order of tasks listed in ``model.new_tasks``. Only needed if `virtual_prompt_style='prompt-tuning'`
   * - **model.prompt_tuning.new_prompt_init_text**
     - list of strings
     - ``[]``
     - The text you want to use for soft prompt initalization if ``model.prompt_tuning.new_prompt_init_methods`` is set to 'text' for a task. Should correspond to the order of tasks listed in ``model.new_tasks``. The text is tokenized and clipped or tiled to match ``total_virtual_tokens`` in ``model.task_templates``. The vocab embeddings associated with each token are copied and use to initialize the soft prompts before tuning.
   * - **model.p_tuning.dropout**
     - float
     - 0.0
     - LSTM prompt encoder dropout prob
   * - **model.p_tuning.num_layers**
     - int
     - 2
     - Num layers in LSTM prompt encoder
   * - **model.tensor_model_parallel_size**
     - int
     - 1
     - intra-layer model parallelism, must match the ``tensor_model_parallel_size`` of the GPT model given at ``language_model_path``
   * - **model.batch_size**
     - int
     - 8
     - global batch size 
   * - **model.data.train_ds**
     - list of strings
     - ``[]``
     - list of ``.json`` or ``.jsonl`` training dataset files with json ojects that have the dataset format described above
   * - **model.data.validation_ds**
     - list of strings
     - ``[]``
     - list of ``.json`` or ``.jsonl`` validation dataset files with json ojects that have the dataset format described above
   * - **model.data.add_eos**
     - bool
     - True
     - Whether to add an EOS token at the end of each training example (recommended). 

An example config file can be found at https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml

Setting New Tasks
^^^^^^^^^^^^^^^^^

After you p-tune or prompt-tune your model, you can always go back and p-tune or prompt-tune your model on more tasks without over writting the virtual prompts who've trained already. You can also use a different number of ``total_virtual_tokens`` between each training session as long as tasks ptuned or prompt tuned at the same time have the same number of ``total_virtual_tokens``. For this reason, when you ptune on a new task, you need to tell your model which of your tasks are new and which ones already exist (and thus you don't want to tune them). You do this by setting the ``new_tasks`` and ``existing_tasks`` values in the config file. 

Example Multi-Task Prompt Tuning Command
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

Example Multi-Task P-Tuning Command After Prompt-Tuning
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


Example Multi-Task Inference 
^^^^^^^^^^
The inference file can contain a mix of prompts from all the tasks the model has been prompt tuned on. 

.. code::

    python megatron_gpt_eval.py \
            virtual_prompt_model=True \
            model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            prompts=[prompt1,prompt2]
            
Prompts in this case should be a list of dictionary examples similar to the ones used during prompt learning. They should have keys that match the fields specified in the prompt template. Fields can be dropped from the prompt dict and their corresponding section of the prompt template will be automatically removed. 

For example, say the prompt template during p-tuning/prompt-tuning looked like:

.. code::

  '<|VIRTUAL_PROMPT_0|> Context: {context} Question: {question} Answer: {answer}'
  
but you don't want to include the answer field during inference. Just don't include the answer field in the prompt dict like below:

.. code::

  prompts = [
              {"taskname": "squad", "context": "some paragraph", "question": "question related to paragraph"},
              {"taskname": "squad", "context": "another paragraph", "question": "a different question related to paragraph"},
            ]
        
And the dataset class will automatically format your input to have the form:

.. code::

  [
      '<|VIRTUAL_PROMPT_0|> Context: some paragraph Question: question related to paragraph Answer: ',
      '<|VIRTUAL_PROMPT_0|> Context: another paragraph Question: a different question related to paragraph Answer: '
  ]
        
Instead of prompt dicts, you can also pass in a list of string paths to .json files on which you want to run inference. Similarly for all other senarios, just add virtual_prompt_model=True if you're using a p-tuned/prompt-tuned model. 

Example prompt learning script: `NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py>`__.

Example prompt tuned inference script: `NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py>`__.
