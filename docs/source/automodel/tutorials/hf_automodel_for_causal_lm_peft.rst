Supervised Fine-tuning (SFT) LLMs with Nemo AutoModel: A Day-0 Support Approach
=========================================================================================

This tutorial explains how to perform Supervised Fine-Tuning (SFT) on Large Language Models (LLMs) using the Nemo framework.
The seamless integration with 🤗 Transformers' ``HFAutoModelForCausalLM`` enables day-0 support for a wide range of models without requiring checkpoint conversions.
For the data aspect of the task, we will show how to load a dataset and format its prompts according to a template.


Introduction
------------

This script demonstrates fine-tuning for a question-answering task using the SQuAD dataset.
It leverages Nemo's training infrastructure and the flexibility of 🤗 Transformers' ``HFAutoModel`` for model loading.

Code Breakdown
--------------

1. **Imports:**

.. code-block:: python
  :linenos:

  import fiddle as fdl
  from lightning.pytorch.loggers import WandbLogger

  from nemo import lightning as nl
  from nemo.collections import llm
  from nemo.lightning import NeMoLogger
  from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform

These imports bring in necessary libraries:

* ``fiddle``: For configuration management.
* ``lightning.pytorch.loggers.WandbLogger``: For logging training metrics to Weights & Biases (wandb).
* ``nemo.lightning``: Nemo's Lightning integration for training.
* ``nemo.collections.llm``: Nemo's LLM-specific functionalities.
* ``nemo.lightning.NeMoLogger``: Nemo's logger for saving training artifacts.
* ``nemo.lightning.pytorch.callbacks.JitConfig, JitTransform``: For using TorchScript JIT compilation.

2. **Data Preparation (``make_squad_hf_dataset``):**

This function prepares the SQuAD dataset for fine-tuning. It uses ``HFDatasetDataModule`` for easy data loading and a formatting function to structure the data into prompts and responses.

.. code-block:: python
  :linenos:
  :emphasize-lines: 5-27

  def make_squad_hf_dataset(tokenizer):
      EOS_TOKEN = tokenizer.eos_token

      def formatting_prompts_func(examples):
          alpaca_prompt = """Below is an instruction...""" # Omitted for brevity
          instruction = examples["context"]
          input = examples["question"]
          output = examples["answers"]['text']
          if isinstance(output, list):
              output = output[0]
          text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
          ans = tokenizer(text)
          ans['labels'] = list(ans['input_ids'][1:]) # Labels shifted by one for next token prediction
          ans['input_ids'] = ans['input_ids'][:-1] # Remove the last token from input_ids
          ans['attention_mask'] = ans['attention_mask'][:-1] # Remove the last token from attention mask
          return ans

      tokenizer = getattr(tokenizer, 'tokenizer', tokenizer) # Handles cases when tokenizer is wrapped
      datamodule = llm.HFDatasetDataModule("rajpurkar/squad", split="train[:100]", pad_token_id=tokenizer.eos_token_id)
      datamodule.map(
          formatting_prompts_func,
          batched=False,
          batch_size=2,
          remove_columns=["id", "title", "context", "question", 'answers'],
      )
      return datamodule

The ``formatting_prompts_func`` takes a dictionary of Squad examples and reformats it into a prompt suitable for instruction tuning. It constructs a prompt using the context as the instruction and the question as the input, and the answer as the target output. Key operations within this function include:

* Formatting input data into a prompt template.
* Tokenizing the formatted text using the provided tokenizer.
* Creating labels by shifting the input IDs by one position to the left, which is standard for language modeling tasks (predicting the next token).
* Removing the last token from input_ids and attention mask to align with the shifted labels.

The ``getattr(tokenizer, 'tokenizer', tokenizer)`` line handles cases where the tokenizer might be wrapped in another object (like a FastTokenizer).

3. **Main Function (``main``):**

This function orchestrates the fine-tuning process.

.. code-block:: python
  :linenos:
  :emphasize-lines: 19, 44, 47, 50-53, 64-70, 77-83

  def main():
    import argparse

    parser = argparse.ArgumentParser()
    # ... Argument parsing ...
    args = parser.parse_args()

    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model) # Day 0 Support!

    wandb = None
    if args.wandb_project is not None:
    # ... Wandb setup ...

    grad_clip = 0.5 # Gradient clipping value

    callbacks = []
    if args.use_torch_jit:
    jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': True}, use_thunder=False)
    callbacks = [JitTransform(jit_config)]

    llm.api.finetune(
        model=llm.HFAutoModelForCausalLM(args.model), # Day 0 Support!
        data=make_squad_hf_dataset(tokenizer.tokenizer),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0, # disable validation
            num_sanity_val_steps=0, # disable sanity check
            accumulate_grad_batches=10, # Accumulate gradients for smaller effective batch size
            gradient_clip_val=grad_clip,
            use_distributed_sampler=False,
            logger=wandb,
            callbacks=callbacks,
            precision="bf16",
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=NeMoLogger(log_dir=args.ckpt_folder, use_datetime_version=False),
        peft=llm.peft.LoRA( # Use LoRA
            target_modules=['*_proj'],
            dim=8,
        ),
    )

The emphasized lines highlight:

* Line 19: Argument parsing for command-line options.
* Line 44: Tokenizer initialization using ``HFAutoModelForCausalLM.configure_tokenizer``.
* Line 47: Gradient clipping setup, disabled when using FSDP.
* Lines 50-53: Setting up JIT compilation if the corresponding flag is passed.
* Lines 64-70: Trainer configuration with important parameters such as gradient accumulation, gradient clipping, precision, and logging.
* Lines 77-83: Using LoRA for Parameter Efficient Fine-Tuning, only training a small subset of the model's parameters.

Running the Script
------------------

1. **Install Nemo:** Follow the official Nemo installation instructions.
2. **Save the script:** Save the code as a Python file (e.g., ``fine_tune.py``).
3. **Run:**

.. code-block:: bash

  python fine_tune.py --model <model_name> --max_steps <num_steps>

# Example:

.. code-block:: bash

  python fine_tune.py --model meta-llama/Llama-2-7b-chat-hf --max_steps 100

Key Advantages
--------------

* **Day-0 Support:** The use of ``HFAutoModelForCausalLM`` provides immediate compatibility with new models released on 🤗 Transformers.
* **No Checkpoint Conversion:** Avoids the hassle of manual checkpoint conversions.
* **Efficient Fine-tuning:** Utilizes LoRA for parameter-efficient adaptation.

Conclusion
----------

This tutorial demonstrated a streamlined approach to fine-tuning LLMs using Nemo and 🤗 Transformers,
emphasizing the ease of use and day-0 support for various models. This setup simplifies the process of adapting cutting-edge LLMs for specific tasks.
