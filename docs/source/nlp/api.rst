NeMo Megatron API
=======================

Pretraining Model Classes
-------------------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel
    :show-inheritance:
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup, on_save_checkpoint, on_load_checkpoint

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_bert_model.MegatronBertModel
    :show-inheritance: 
    :no-members:
    :members: training_step, validation_step, build_train_valid_test_datasets, build_LDDL_data, setup, on_save_checkpoint, on_load_checkpoint

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_retrieval_model.MegatronRetrievalModel
    :show-inheritance: 
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model
    :show-inheritance: 
    :no-members:
    :members: complete, encode, decode, add_special_tokens_to_tokenizer, training_step, validation_step, build_train_valid_test_datasets, setup

Customization Model Classes
---------------------------

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model.MegatronGPTSFTModel
    :show-inheritance: 
    :no-members:
    :members: generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_adapter_model.MegatronGPTAdapterLearningModel
    :show-inheritance: 
    :no-members:
    :members: __init__, state_dict, generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_adapter_model.MegatronGPTInfusedAdapterModel
    :show-inheritance: 
    :no-members:
    :members: __init__, state_dict, generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model.MegatronGPTPromptLearningModel
    :show-inheritance: 
    :no-members:
    :members: built_virtual_prompt_dataset, generate, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model.MegatronT5AdapterLearningModel
    :show-inheritance: 
    :no-members:
    :members: __init__, state_dict, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model.MegatronT5AdapterLearningModel
    :show-inheritance: 
    :no-members:
    :members: _add_adapters_to_component, __init__, state_dict, training_step, validation_step, build_train_valid_test_datasets, setup

.. autoclass:: nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model.MegatronT5InfusedAdapterModel
    :show-inheritance: 
    :no-members:
    :members: _add_adapters_to_component, __init__, state_dict, training_step, validation_step, build_train_valid_test_datasets, setup

Modules
-------


Datasets
--------

