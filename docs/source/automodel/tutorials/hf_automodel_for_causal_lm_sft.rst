Parameter Efficient Fine-tuning (PEFT) LLMs with Nemo AutoModel: A Day-0 Support Approach
===================================================================================================

This tutorial explains how to fine-tune a large language model (LLM) on the SQuAD dataset using the 🤗 Transformers library and NVIDIA's Nemo framework. A key feature of this approach is its *day0 support* for any Hugging Face (HF) ``AutoModelForCausalLM`` model, eliminating the need for checkpoint conversions.

Understanding the Code
-----------------------

The Python code implements a script for fine-tuning. Let's break it down in detail:

1. **Imports:** Imports necessary libraries:

- ``fiddle as fdl``: For configuration management, particularly for optimizers.
- ``lightning.pytorch as pl``: PyTorch Lightning for streamlined training.
- ``torch.utils.data.DataLoader``: PyTorch's data loading utility.
- ``nemo.lightning as nl``: Nemo's Lightning integration.
- ``nemo.collections.llm``: Nemo's LLM components.
- Other imports related to Transformer Engine (TE) acceleration and callbacks.

2. ``SquadDataModuleWithPthDataloader`` **Class:** This class inherits from ``nemo.collections.llm.SquadDataModule`` and overrides the ``_create_dataloader`` method. This customization allows explicit control over the PyTorch ``DataLoader`` instantiation, setting parameters like ``num_workers``, ``pin_memory``, and ``batch_size``. This is useful for performance tuning.

.. code-block:: python
  :linenos:

  class SquadDataModuleWithPthDataloader(llm.SquadDataModule):
    def _create_dataloader(self, dataset, mode, **kwargs) -> DataLoader:
      return DataLoader(
        dataset,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        persistent_workers=self.persistent_workers,
        collate_fn=dataset.collate_fn,
        batch_size=self.micro_batch_size,
        **kwargs,
      )

3. ``squad`` **Function:** This function takes a tokenizer as input and creates an instance of the ``SquadDataModuleWithPthDataloader`` class. It configures the data module with:

- ``tokenizer``: The tokenizer used for text processing.
- ``seq_length``: The maximum sequence length for input text.
- ``micro_batch_size``: The batch size per GPU.
- ``global_batch_size``: The total batch size across all GPUs.
- Other dataset-specific keyword arguments.

.. code-block:: python
  :linenos:

  def squad(tokenizer) -> pl.LightningDataModule:
    return SquadDataModuleWithPthDataloader(
      tokenizer=tokenizer,
      seq_length=512,
      micro_batch_size=2,
      global_batch_size=128, # assert gbs == mbs * accumulate_grad_batches
      num_workers=0,
      dataset_kwargs={
        "sanity_check_dist_workers": False,
        "pad_to_max_length": True,
        "get_attention_mask_from_fusion": True,
      },
    )

4. ``main`` **Function:** This is the core training script:

- **Argument Parsing:** Uses ``argparse`` to handle command-line arguments, including the model name (``--model``), training strategy (``--strategy``), number of devices (``--devices``), maximum training steps (``--max-steps``), and other options.
- **Wandb Setup:** Initializes Wandb logging if the ``--wandb-project`` argument is provided.
- **Model Instantiation:** Creates the LLM using ``llm.HFAutoModelForCausalLM(model_name=args.model, model_accelerator=model_accelerator)``. This is where the HF model is loaded. The ``model_accelerator`` argument allows to use Transformer Engine for acceleration. This demonstrates the *day0 support* for HF models.
- **Callbacks:** Sets up callbacks for the training loop. In this case, it adds a ``JitTransform`` callback if ``--use-torch-jit`` is specified, which uses TorchScript to optimize the model.
- **Fine-tuning:** Calls ``llm.api.finetune`` to start the fine-tuning process. This function takes the model, data module, trainer, optimizer, and other configuration parameters.
- The ``trainer`` is an instance of ``nemo.lightning.Trainer``, which handles the training loop, logging, and other training-related tasks.
- The ``optim`` argument is built using ``fdl.build`` and configures the optimizer (Adam with a flat learning rate schedule).
- **TE Check and Model Saving:** After training, it checks if Transformer Engine acceleration was successful and saves the model if a save path is provided.

Code Example
------------

.. code-block:: python
  :linenos:

  # ... (Imports and class/function definitions as before)

  def main():
  # ... (Argument parsing)

  model = llm.HFAutoModelForCausalLM(model_name=args.model, model_accelerator=model_accelerator) # HF model loading
  tokenizer = model.tokenizer

  callbacks = []
  if args.use_torch_jit:
  # ... (Jit configuration)
    callbacks = [JitTransform(jit_config)]

  llm.api.finetune(
    model=model,
    data=squad(tokenizer),
    trainer=nl.Trainer(
        # ... (Trainer configuration, including devices, strategy, etc.)
    ),
    optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)), # Optimizer definition
    log=None,
  )


  if __name__ == '__main__':
    main()

Modifying the Script
--------------------

- **Model Selection:** Change the ``--model`` argument to use a different HF ``AutoModelForCausalLM`` model (e.g., ``google/flan-t5-xl``).
- **Training Configuration:**
- ``--max_steps``: Number of training steps.
- ``--devices``: Number of GPUs to use.
- ``--strategy``: Training strategy (``ddp``, ``fsdp``, etc.).
- Adjust batch sizes in the ``squad`` function.
- **Logging and Saving:**
- ``--wandb-project``: Enable Wandb logging.
- ``--model-save-path``: Specify a path to save the fine-tuned model.
- **Optimizer:** Modify the optimizer configuration in the ``llm.api.finetune`` call.

Running the Script
------------------

1. **Install Dependencies:** ``pip install nemo transformers pytorch-lightning``
2. **Save the Script:** Save the code as ``squad_finetuning.py``.
3. **Run:**

.. code-block:: bash

  python squad_finetuning.py \
    --model <model_name> \
    --devices <num_gpus> \
    --max_steps <training_steps>

Example:

.. code-block:: bash

  python squad_finetuning.py \
    --model meta-llama/Llama-3.2-1B \
    --devices 1 \
    --max_steps 100 \
    --wandb-project my-squad-project \
    --model-save-path ./my_fine_tuned_model

Key Advantages
--------------

- **Day0 HF Support:** No checkpoint conversions needed.
- **Flexibility:** Easily adaptable to different LLMs and training configurations.
- **Simplified Fine-tuning:** Streamlined process with Nemo and 🤗 Transformers.
