=========================================================
Guide for running optimized SFT / PEFT in NeMo for Gemma
=========================================================

Tuning with a Packed SFT Dataset for Gemma Models
=================================================
This section provides detailed steps to prepare a packed Sequence-to-Sequence Fine-Tuning (SFT) dataset for Gemma (GPT-like) models, using the example of the "dolly" dataset. Although we focus on "dolly", the methodology should be applicable to any dataset. This provide a significant boost to SFT / PEFT performance in NeMo.

Pre-requisites
@@@@@@@@@@@@@@

Scripts are provided at: ``/workspace/sequence_packing``

0. Download and Split Data
##########################
First, download and prepare your dataset by splitting it into train, validation, and test sets. See data preparation section below in SFT for preparing "dolly" dataset.

1. Tokenize Dataset
###################
Tokenize the entire dataset to obtain tokenized sequences, represented by indices.

.. code-block:: bash

   python tokenize_dataset.py \
   model.data.train_ds.file_names=[/path/to/training.jsonl] \
   model.data.train_ds.max_seq_length=4096 \
   model.restore_from_path=/path/to/gemma-7b.nemo \
   +output_path=/path/to/my_dataset.npy

.. note::

   - `model.data.train_ds.max_seq_length` is used to truncate long sequences.
   - A full nemo model file is required for simplicity and readability.

2. Group by Length
##################
Group the tokenized sequences by their sequence length.

.. code-block:: bash

   python create_hist.py \
   --input_file=my_dataset.npy \
   [--output_dir=./per_seq_data] \
   [--output_histogram=histogram.npy] \
   [--truncate_seq_len=2048]

3. Run Packing Algorithm
########################
Run the packing algorithm to find an efficient packing strategy.

.. code-block:: bash

   python run_packing.py \
   --output_dir <OUTPUT_DIR> \
   --pack_size <PACK_SIZE> \
   [--input_dir=./per_seq_data] \
   [--histogram=histogram.npy] \
   [--packing_algorithm=first_fit_shuffle] \
   [--seed=0]

4. Train with Packed Sequences
##############################
Enable training with packed sequences by modifying the SFT / PEFT config file. We also need to reduce both micro batch size and global batch size due to packing. Here we set ``global_batch_size=1`` with sequence length 4096.

.. code-block:: bash
   model.data.train_ds.file_names=/path/to/dolly_packed_4096_seed0.npy \
   +model.data.train_ds.packed_sequence=True \
   model.micro_batch_size=1 \
   model.global_batch_size=8 \

Appendix: Complete Example
##########################
.. note:: Internal paths are for reference only and should not be shared externally.

.. code-block:: bash

   python tokenize_dataset.py \
   model.data.train_ds.file_names=[/path/to/datasets/squad/1_squad_train.jsonl] \
   model.data.train_ds.max_seq_length=4096 \
   model.restore_from_path=/path/to/gemma-7b.nemo \
   +output_path=gemma_squad_packed/my_dataset.npy

   python create_hist.py --input_file=gemma_squad_packed/my_dataset.npy
   python run_packing.py --output_dir gemma_squad_packed --pack_size 2048
   python run_packing.py --output_dir gemma_squad_packed --pack_size 4096
   python run_packing.py --output_dir gemma_squad_packed --pack_size 8192

NeMo Framework Supervised fine-tuning (SFT) with Gemma
=======================================================

Project Description
@@@@@@@@@@@@@@@@@@@

Learning Goals
##############
Often we want to adapt or customize foundation models to be more performant on our specific task. Fine-tuning refers to how we can modify the weights of a pre-trained foundation model with additional custom data. Supervised fine-tuning (SFT)
refers to unfreezing all the weights and layers in our model and training on a newly labeled set of examples. We can fine-tune to incorporate new, domain-specific knowledge or teach the foundation model what type of response to provide. One
specific type of SFT is also referred to as “instruction tuning” where we use SFT to teach a model to follow instructions better.

In this project, you’ll test out the supervised finetuning method on the gemma model using an instructive dataset.

NeMo Tools and Resources
########################

#. `NeMo Github repo <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/tuning>`__
#. NeMo Gemma Training container: ``nvcr.io/nvidia/nemo:24.01.gemma``

Software Requirements
#####################

#. Use the latest `NeMo Framework Training container <https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training/tags>`__
#. This readme has been tested using the container: ``nvcr.io/nvidia/nemo:24.01.gemma`` on DGX Cloud. It is expected to work similarly on other environments.


Hardware Requirements
#####################
#. Minimum 8xH100 80G (1 node) for SFT on 7B.
#. However, SFT on all (2B/7B) model sizes can be run on multiple nodes.

Data
####
Databricks-dolly-15k is an open-source dataset of instruction-following records generated by thousands of Databricks employees in several of the behavioral categories outlined in the InstructGPT paper, including brainstorming, classification,
closed QA, generation, information extraction, open QA, and summarization
For more details about the data refer to `databricks-dolly-15k | Hugging Face <https://huggingface.co/datasets/databricks/databricks-dolly-15k>`__

The following steps have been tested with this container: ``nvcr.io/nvidia/nemo:24.01.gemma``

[Optional] Convert Gemma from jax or pytorch format to NeMo format
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

If you already have a .nemo file for gemma models, you can skip this step.

Optional Step: Convert to .nemo
#################################

Run the container using the following command ::

   docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.01.gemma bash

#. Option 1: Convert the jax model to .nemo model ::

   pip install orbax jax flax jaxlib; \
   export PYTHONPATH=/path/to/deepmind/models/gemma_jax/code:$PYTHONPATH; \
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_jax_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/jax/7b \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

#. Option 2: Convert the pytorch model to .nemo model ::

   pip install fairscale==0.4.13 immutabledict==4.1.0 tensorstore==0.1.45; \
   export PYTHONPATH=/path/to/deepmind/models/gemma_pytorch/code:$PYTHONPATH; \
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_pyt_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/pyt/7b.ckpt \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

#. Option 3: Convert the HuggingFace model to .nemo model ::

   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_hf_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/hf/7b \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

The generated gemma-7b.nemo file uses distributed checkpointing and can be loaded with any tensor parallel (tp) or pipeline parallel (pp) combination without reshaping/splitting.

Prepare data
@@@@@@@@@@@@

Step 1: Download dataset
########################
Download the dolly-15k dataset from huggingface ::

   git clone https://huggingface.co/datasets/databricks/databricks-dolly-15k

Once downloaded, check the size of the file (databricks-dolly-15k.jsonl) ::

   $ du -sh databricks-dolly-15k/databricks-dolly-15k.jsonl
   13M     databricks-dolly-15k/databricks-dolly-15k.jsonl

If the sizes do not match, delete the old file and manually copy the download link address and directly wget the file ::

   wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl

Step 2: Data Preprocessing
##########################

#. Next we need to pre-process the data to ensure it’s in the correct format.
#. The expected format is a JSONL file with {‘input’: ‘xxx’, ‘output’: ‘yyy’} pairs.
#. In order to run the pre-processing you will use the `script <https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/master/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py>`__ that has already been prepared for you. Run this script and passing your jsonl file as `--input`. In order to run the script you need to launch the container.

If the container is not already running use the following command ::

   docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.01.gemma bash

And then run the following data preprocess script

.. code-block:: python

   python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py --input databricks-dolly-15k/databricks-dolly-15k.jsonl

Example output ::

   Preprocessing data to jsonl format...
   Data was successfully preprocessed and saved by databricks-dolly-15k/databricks-dolly-15k-output.jsonl .

Check that the output jsonl files exists ::

   $ ls databricks-dolly-15k/
   .git/
   .gitattributes
   README.md
   databricks-dolly-15k-output.jsonl
   databricks-dolly-15k.jsonl

Check the first example in the output jsonl file ::

   $ head -n 1 databricks-dolly-15k/databricks-dolly-15k-output.jsonl
   {"input": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.\n\nWhen did Virgin Australia start operating?", "output": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.", "category": "closed_qa"}


Step 3: Split the data into train, validation and test.
#######################################################

Generate the train, test and validation splits- you may use your own script to do this or create a new script and use the following sample split_train_val.py by copying it over in the databricks-dolly-15k directory ::

   import json
   import random

   input_file = "databricks-dolly-15k-output.jsonl"
   training_output_file = "training.jsonl"
   validation_output_file = "validation.jsonl"
   test_output_file = "test.jsonl"

   # Specify the proportion of data for training and validation
   train_proportion = 0.80
   validation_proportion = 0.15
   test_proportion = 0.05

   # Read the JSONL file and shuffle the JSON objects
   with open(input_file, "r") as f:
       lines = f.readlines()
       random.shuffle(lines)

   # Calculate split indices
   total_lines = len(lines)
   train_index = int(total_lines * train_proportion)
   val_index = int(total_lines * validation_proportion)

   # Distribute JSON objects into training and validation sets
   train_data = lines[:train_index]
   validation_data = lines[train_index:train_index+val_index]
   test_data = lines[train_index+val_index:]

   # Write JSON objects to training file
   with open(training_output_file, "w") as f:
       for line in train_data:
           f.write(line.strip() + "\n")

   # Write JSON objects to validation file
   with open(validation_output_file, "w") as f:
       for line in validation_data:
           f.write(line.strip() + "\n")

   # Write JSON objects to training file
   with open(test_output_file, "w") as f:
       for line in test_data:
           f.write(line.strip() + "\n")

Then go to the ``databricks-dolly-15k`` directory and generate the splits::

   python3 split_train_val.py

Check for the train, test and validation jsonl files ::

   $ ls
   README.md
   databricks-dolly-15k.jsonl
   databricks-dolly-15k-output.jsonl
   split_train_val.py
   training.jsonl
   validation.jsonl
   test.jsonl

Step 4: Run the SFT finetuning script
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Set the environment variables, pass the paths to your train, test, and validation data files ::

   MODEL="YOUR PATH TO gemma-7b.nemo"
   TRAIN="[YOUR PATH TO databricks-dolly-15k/train.jsonl]"
   VALID="[YOUR PATH TO databricks-dolly-15k/validation.jsonl]"
   TEST="[YOUR PATH TO databricks-dolly-15k/test.jsonl]"
   VALID_NAMES="[databricks-dolly-15k]"


Set the concat sampling probability. This depends on the number of files being passed in the train set and how much percentage of the fine tuning data would you like to use from each file. Note sum of concat sampling probabilities should be 1.0.
For example, the following is an example for setting concat sampling probability for a train set with 2 jsonl files. ::

   TRAIN="[/path/to/dataset_1.jsonl,/path/to/dataset_2.jsonl]"
   CONCAT_SAMPLING_PROBS="[0.3,0.7]"

In our example we are using 1 train file so ``CONCAT_SAMPLING_PROBS="[1.0]"``
Set the tensor parallelism and pipeline parallelism values based on the model you are using. ::

   CONCAT_SAMPLING_PROBS="[1]"
   TP_SIZE=2
   PP_SIZE=1

Run the SFT command by appropriately setting the values for the parameters such as the number of steps, model checkpoint path, batch sizes etc. For a full reference of parameter settings refer to the `config file <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml>`__ ::

   torchrun --nproc_per_node=8 \
   /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
      trainer.precision=bf16 \
      trainer.devices=8 \
      trainer.num_nodes=1 \
      trainer.val_check_interval=0.1 \
      trainer.max_steps=50 \
      model.restore_from_path=${MODEL} \
      model.peft.peft_scheme=none \
      model.micro_batch_size=1 \
      model.global_batch_size=128 \
      model.tensor_model_parallel_size=${TP_SIZE} \
      model.pipeline_model_parallel_size=${PP_SIZE} \
      model.megatron_amp_O2=True \
      model.sequence_parallel=True \
      model.activations_checkpoint_granularity=selective \
      model.activations_checkpoint_method=uniform \
      model.optim.name=distributed_fused_adam \
      model.optim.lr=5e-6 \
      model.answer_only_loss=True \
      model.data.train_ds.file_names=${TRAIN_DS} \
      model.data.validation_ds.file_names=${VALID_DS} \
      model.data.test_ds.file_names=${TEST_DS} \
      model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
      model.data.train_ds.max_seq_length=4096 \
      model.data.validation_ds.max_seq_length=4096 \
      model.data.train_ds.micro_batch_size=1 \
      model.data.train_ds.global_batch_size=128 \
      model.data.validation_ds.micro_batch_size=1 \
      model.data.validation_ds.global_batch_size=128 \
      model.data.test_ds.micro_batch_size=1 \
      model.data.test_ds.global_batch_size=256 \
      model.data.train_ds.num_workers=0 \
      model.data.validation_ds.num_workers=0 \
      model.data.test_ds.num_workers=0 \
      model.data.validation_ds.metric.name=loss \
      model.data.test_ds.metric.name=loss \
      exp_manager.create_wandb_logger=False \
      exp_manager.explicit_log_dir=/results \
      exp_manager.resume_if_exists=True \
      exp_manager.resume_ignore_no_checkpoint=True \
      exp_manager.create_checkpoint_callback=True \
      exp_manager.checkpoint_callback_params.monitor=validation_loss \
      exp_manager.checkpoint_callback_params.save_best_model=False \
      exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
      ++cluster_type=BCP \
      model.sequence_parallel=True \
	  ++model.apply_rope_fusion=True \
	  ++model.optim.overlap_grad_sync=True \
      ++model.optim.overlap_param_sync=True \
      ++model.optim.contiguous_grad_buffer=True \
      ++model.optim.grad_sync_dtype=bf16 \
	  ++model.fp8=True \
	  ++model.fp8_e4m3=False \
	  ++model.fp8_hybrid=True \
	  ++model.fp8_margin=0 \
	  ++model.fp8_interval=1 \
	  ++model.fp8_amax_history_len=128 \
	  ++model.fp8_amax_compute_algo=max

Note: For running SFT on multiple nodes (for example on a Slurm cluster, replace the ``torchrun --nproc_per_node=8`` with ``python``.

Step 6: Run evaluation
######################
Run evaluation using `megatron_gpt_peft_eval.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py>`__

Set the appropriate model checkpoint path, test file path, batch sizes, number of tokens etc. and run evaluation on the test file ::

  PATH_TO_TRAINED_MODEL=/results/megatron_gpt_peft_none_tuning/checkpoints/megatron_gpt_peft_none_tuning.nemo
  TEST_DS="[YOUR PATH TO test.jsonl]"
  python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
      model.restore_from_path=${PATH_TO_TRAINED_MODEL} \
      trainer.devices=8 \
      model.data.test_ds.file_names=${TEST_DS} \
      model.data.test_ds.names=['dolly-15k_test'] \
      model.data.test_ds.global_batch_size=16 \
      model.data.test_ds.micro_batch_size=2 \
      model.data.test_ds.tokens_to_generate=20 \
      model.tensor_model_parallel_size=1 \
      model.pipeline_model_parallel_size=1 \
      inference.greedy=True \
      model.data.test_ds.output_file_path_prefix=/results/sft_results \
      model.data.test_ds.write_predictions_to_file=True


Sample Output ::

   $ tail -n 4 sft_results.jsonl

   {"sentence": "What is Azure HDInsight? Azure HDInsight is a cloud service that provides a high-performance, scalable, and cost-effective way to run Apache Hadoop on the"}
   {"sentence": "What is carnitine? Carnitine is a fatty acid that is found in the body. It is used to produce energy in the mitochondria of the cells. Carnit"}
   {"sentence": "List some TV shows that Canadian actor William B. Davis has been in."}
   {"sentence": "Identify which instrument is string or percussion: Handbell, Dobro, Drum"}

Note, This is only a sample output (based of a toy SFT example) and your output may vary. The performance can be further improved by fine tuning the model for more steps.

NeMo Framework PEFT with Gemma
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Project Description
@@@@@@@@@@@@@@@@@@@

Learning Goals
##############

This project aims to demonstrate how to adapt or customize foundation models to improve performance on specific tasks.

This optimization process is known as fine-tuning, which involves adjusting the weights of a pre-trained foundation model with custom data.

Considering that foundation models can be significantly large, a variant of fine-tuning has gained traction recently, known as parameter-efficient fine-tuning (PEFT). PEFT encompasses several methods, including P-Tuning, LoRA, Adapters, and IA3.

For those interested in a deeper understanding of these methods, we have included a list of additional resources at the end of this document.

This project involves applying various fine-tuning methods to gemma model. In this readme you will implement and evaluate several parameter-efficient fine-tuning methods using a domain and task specific dataset. This readme has been tested for P-Tuning and LoRA.


NeMo Tools and Resources
########################

#. `NeMo Github repo <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/tuning>`__

Educational Resources
#####################

#. Blog: Understanding `Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters <https://lightning.ai/pages/community/article/understanding-llama-adapters/>`__
#. NeMo documentation: `Introduction to P-tuning <https://llm.ngc.nvidia.com/docs/model-customization-with-p-tuning.html#model-customization-with-p-tuning>`__
#. NeMo notebook/tutorial: `Introduction to p-tuning and prompt-tuning <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Multitask_Prompt_and_PTuning.ipynb>`__


Software Requirements
#####################

#. Use the latest `NeMo Framework Training container <https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training/tags>`__
#. This readme has been tested on: ``nvcr.io/nvidia/nemo:24.01.gemma``


Hardware Requirements
#####################
#. Minimum 1xH100 80G for PEFT on 7B. This readme has been tested on 8xH100 80G.

[Optional] Convert Gemma from jax or pytorch format to NeMo format
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

If you already have a .nemo file for gemma models, you can skip this step.

Optional Step: Convert to .nemo
#################################

Run the container using the following command ::

   docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.01.gemma bash

#. Option 1: Convert the jax model to .nemo model ::

   pip install orbax jax flax jaxlib; \
   export PYTHONPATH=/path/to/deepmind/models/gemma_jax/code:$PYTHONPATH; \
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_jax_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/jax/7b \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

#. Option 2: Convert the pytorch model to .nemo model ::

   pip install fairscale==0.4.13 immutabledict==4.1.0 tensorstore==0.1.45; \
   export PYTHONPATH=/path/to/deepmind/models/gemma_pytorch/code:$PYTHONPATH; \
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_pyt_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/pyt/7b.ckpt \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

#. Option 3: Convert the HuggingFace model to .nemo model ::

   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_hf_to_nemo.py \
    --input_name_or_path /path/to/gemma/checkpoints/hf/7b \
    --output_path /path/to/gemma-7b.nemo \
    --tokenizer_path /path/to/tokenizer.model

The generated gemma-7b.nemo file uses distributed checkpointing and can be loaded with any tensor parallel (tp) or pipeline parallel (pp) combination without reshaping/splitting.

Prepare data
@@@@@@@@@@@@

Step 1: Download the PubMedQA dataset and run the split_dataset.py script in the cloned directory.
##################################################################################################

Download the dataset ::

    git clone https://github.com/pubmedqa/pubmedqa.git
    cd pubmedqa
    cd preprocess
    python split_dataset.py pqal

After running the split_dataset.py script, you will see the test_set as well as ten different directories which each contains a different train/validation fold ::

    $ cd ../..
    $ ls pubmedqa/data/
    ori_pqal.json
    pqal_fold0
    pqal_fold1
    pqal_fold2
    pqal_fold3
    pqal_fold4
    pqal_fold5
    pqal_fold6
    pqal_fold7
    pqal_fold8
    pqal_fold9
    test_ground_truth.json
    test_set.json

Below is an example of what the objects look like inside of the PubMedQA train, validation and test splits ::

    "18251357": {
        "QUESTION": "Does histologic chorioamnionitis correspond to clinical chorioamnionitis?",
        "CONTEXTS": [
            "To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother.",
            "A retrospective review was performed on 52 cases with a histologic diagnosis of acute chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as controls. Cases and controls were selected sequentially. Maternal medical records were reviewed for indicators of maternal infection.",
            "Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = 0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more clinical indicators was significantly associated with the presence of histologic chorioamnionitis (p = 0.019)."
        ],
        "reasoning_required_pred": "yes",
        "reasoning_free_pred": "yes",
        "final_decision": "yes",
        "LONG_ANSWER": "Histologic chorioamnionitis is a reliable indicator of infection whether or not it is clinically apparent."
    },


Step 2: Data Preprocessing
##########################

Use the below script to convert the train/valid/test PubMedQA data into the JSONL format that NeMo needs for PEFT. In this example, we have named the below script, “preprocess_to_jsonl.py”, and placed it inside of the pubmedqa repository we have previously cloned.

.. code-block:: python

    import json
    def read_jsonl (fname):
        obj = []
        with open(fname, 'rt') as f:
            st = f.readline()
            while st:
                obj.append(json.loads(st))
                st = f.readline()
        return obj
    def write_jsonl(fname, json_objs):
        with open(fname, 'wt') as f:
            for o in json_objs:
                f.write(json.dumps(o)+"\n")
    def form_question(obj):
        st = ""
        st += f"QUESTION: {obj['QUESTION']}\n"
        st += "CONTEXT: "
        for i, label in enumerate(obj['LABELS']):
            st += f"{obj['CONTEXTS'][i]}\n"
        st += f"TARGET: the answer to the question given the context is (yes|no|maybe): "
        return st
    def convert_to_jsonl(data_path, output_path):
        data = json.load(open(data_path, 'rt'))
        json_objs = []
        for k in data.keys():
            obj = data[k]
            prompt = form_question(obj)
            completion = obj['reasoning_required_pred']
            json_objs.append({"input": prompt, "output": completion})
        write_jsonl(output_path, json_objs)
        return json_objs
    def main():
        test_json_objs = convert_to_jsonl("data/test_set.json", "pubmedqa_test.jsonl")
        train_json_objs = convert_to_jsonl("data/pqal_fold0/train_set.json", "pubmedqa_train.jsonl")
        dev_json_objs = convert_to_jsonl("data/pqal_fold0/dev_set.json", "pubmedqa_val.jsonl")
        return test_json_objs, train_json_objs, dev_json_objs
    if __name__ == "__main__":
        main()

After running the above script, you will see the ``pubmedqa_train.jsonl``, ``pubmedqa_val.jsonl``, ``pubmedqa_test.jsonl`` files appear in the directory you copied and ran the preprocessing script ::

    $ ls pubmedqa/
    data
    evaluation.py
    exp
    get_human_performance.py
    LICENSE
    nemo_preprocess.py
    preprocess
    pubmedqa_test.jsonl
    pubmedqa_train.jsonl
    pubmedqa_val.jsonl
    README.md

Below is what the formatting will look like once we have used the above script for converting the PubMedQA data into the format that NeMo expects for SFT and PEFT ::

    {"input": "QUESTION: Failed IUD insertions in community practice: an under-recognized problem?\nCONTEXT: The data analysis was conducted to describe the rate of unsuccessful copper T380A intrauterine device (IUD) insertions among women using the IUD for emergency contraception (EC) at community family planning clinics in Utah.\n...",
    "output": "yes"}


Run the PEFT finetuning script
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Step 1: Set the experiment configs
##################################

The megatron_gpt_peft_tuning_config.yaml file is used to configure the parameters for the running PEFT training jobs in NeMo with P-Tuning and LoRA techniques for language model tuning.
Set the environment variables, pass the paths to your train, test and validation data files ::

   MODEL="YOUR PATH TO gemma-7b.nemo"
   TRAIN_DS="[YOUR PATH TO pubmedqa/pubmedqa_train.jsonl]"
   VALID_DS="[YOUR PATH TO pubmedqa/pubmedqa_val.jsonl]"
   TEST_DS="[YOUR PATH TO pubmedqa/pubmedqa_test.jsonl]"
   TEST_NAMES="[pubmedqa]"
   SCHEME="lora"


Set the concat sampling probability. This depends on the number of files being passed in the train set and how much percentage of the fine tuning data would you like to use from each file. Note sum of concat sampling probabilities should be 1.0.
For example, the following is an example for setting concat sampling probability for a train set with 2 jsonl files ::

   TRAIN_DS="[/path/to/dataset_1.jsonl,/path/to/dataset_2.jsonl]"
   CONCAT_SAMPLING_PROBS="[0.3,0.7]"

In our example we are using 1 train file so ``CONCAT_SAMPLING_PROBS="[1.0]"``

Step 2: Run PEFT training
#########################

Run the PEFT command by appropriately setting the values for the parameters such as the number of steps, model checkpoint path, batch sizes etc. For a full reference of parameter settings refer to the `config <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml>`__ file::

    torchrun --nproc_per_node=8 \
    /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=8 \
        trainer.num_nodes=1 \
        trainer.precision=bf16 \
        trainer.val_check_interval=20 \
        trainer.max_steps=50 \
        model.megatron_amp_O2=False \
        ++model.mcore_gpt=True \
        model.tensor_model_parallel_size=${TP_SIZE} \
        model.pipeline_model_parallel_size=${PP_SIZE} \
        model.micro_batch_size=1 \
        model.global_batch_size=8 \
        model.restore_from_path=${MODEL} \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.validation_ds.file_names=${VALID_DS} \
        model.peft.peft_scheme=${SCHEME} \
        exp_manager.explicit_log_dir=/results \
        ++model.fp8=True \
	    ++model.fp8_e4m3=False \
	    ++model.fp8_hybrid=True \
	    ++model.fp8_margin=0 \
	    ++model.fp8_interval=1 \
	    ++model.fp8_amax_history_len=128 \
	    ++model.fp8_amax_compute_algo=max \
        ++model.fp8_params=True

Set ``$SCHEME="ptuning"`` for ptuning instead of lora.

Step 3: Run inference
#####################

#. Set model.restore_from_path to the path for the gemma-7b.nemo model.
#. Set model.peft.restore_from_path to the path for the PEFT checkpoint that will be saved inside of your experiment directory.
#. Set model.test_ds.file_names to the path of the pubmedqa_test.jsonl file

Please configure ``$tokens_to_generate`` and ``output_file_path_prefix`` according to your project needs ::
    PATH_TO_TRAINED_MODEL=/results/megatron_gpt_peft_lora_tuning/checkpoints/megatron_gpt_peft_lora_tuning.nemo
    python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
        model.restore_from_path=${MODEL} \
        model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
        trainer.devices=8 \
        model.data.test_ds.file_names=${TEST_DS} \
        model.data.test_ds.names=${TEST_NAMES} \
        model.data.test_ds.global_batch_size=32 \
        model.data.test_ds.micro_batch_size=4 \
        model.data.test_ds.tokens_to_generate=20 \
        inference.greedy=True \
        model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
        model.data.test_ds.write_predictions_to_file=True

Step 4 (Optional): Merge LORA weights
######################################

If needed, you can merge LORA weights into a base GPT LM. Currently, only PP=1 is supported.

.. code-block:: bash

    PATH_TO_MERGED_MODEL=/results/megatron_gpt_peft_lora_tuning/checkpoints/megatron_gpt_lora_merged.nemo
    python /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py \
        trainer.accelerator=gpu \  # Use 'cpu' if the model cannot fit in memory
        tensor_model_parallel_size=${TP_SIZE} \
        pipeline_model_parallel_size=1 \
        gpt_model_file=${MODEL} \
        lora_model_path=${PATH_TO_TRAINED_MODEL} \
        merged_model_path=${PATH_TO_MERGED_MODEL}


To find the TP of the LORA checkpoint, you can visually examine the output of:

.. code-block:: bash

    tar -tvf ${PATH_TO_MERGED_MODEL}

Replace `${PATH_TO_MERGED_MODEL}` with the path to your merged model checkpoint.


