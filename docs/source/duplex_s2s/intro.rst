Duplex Speech-to-Speech (S2S)
================================

.. note::
   The Duplex Speech-to-Speech collection is still in active development and the code is likely to change.

Duplex Speech-to-Speech (S2S) refers to a collection that augments pre-trained Large Language Models (LLMs) with speech understanding and generation capabilities. This enables seamless end-to-end speech-to-speech conversation systems that can understand spoken input and produce natural-sounding spoken responses.

The key components of this collection include:
1. Audio perception modules that convert speech input into representations compatible with LLMs
2. Language model components that process these representations and generate appropriate responses
3. Speech generation modules that convert text outputs back into speech

The collection is designed to be compact, efficient, and to support easy swapping of different LLMs. It has been successfully tested with various models including TinyLlama, Llama3, Gemma2, and Qwen2.5.

This collection supports multiple model architectures such as DuplexS2SModel, DuplexS2SSpeechDecoderModel, and SALM (Speech-Augmented Language Model).

Using Pretrained Models
----------------------

After :ref:`installing NeMo<installation>`, you can load and use a pretrained duplex_s2s model as follows:

.. code-block:: python

    import nemo.collections.duplex_s2s as nemo_duplex_s2s
    
    # Load a pretrained SALM model
    model = nemo_duplex_s2s.models.SALM.from_pretrained("path/to/pretrained_checkpoint")
    
    # Alternatively, loading from HuggingFace Hub (when available)
    # model = nemo_duplex_s2s.models.SALM.from_pretrained("vendor/model_name")
    
    # Set model to evaluation mode
    model = model.eval()

Inference with Pretrained Models
--------------------------------

You can run inference using the loaded pretrained model:

.. code-block:: python

    import torch
    import torchaudio
    
    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:  # Most models expect 16kHz audio
        audio_signal = torchaudio.functional.resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000
    
    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)
    
    # Create a prompt for SALM model inference
    # The audio_locator_tag is a special token that will be replaced with audio embeddings
    prompt = [{"role": "user", "content": f"{model.audio_locator_tag}"}]
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            prompts=[prompt],
            audios=audio_signal,
            audio_lens=audio_len,
            generation_config=None  # You can customize generation parameters here
        )
    
    # Process the output tokens
    response = model.tokenizer.ids_to_text(output[0])
    print(f"Model response: {response}")

Training a Model
---------------

This example demonstrates how to train a SALM model:

.. code-block:: python

    from omegaconf import OmegaConf
    import torch
    from pytorch_lightning import Trainer
    
    import nemo.collections.duplex_s2s as nemo_duplex_s2s
    from nemo.collections.duplex_s2s.data import SALMDataset, DataModule
    from nemo.utils.exp_manager import exp_manager
    
    # Load configuration
    config_path = "path/to/config.yaml"  # E.g., from examples/duplex_s2s/conf/salm.yaml
    cfg = OmegaConf.load(config_path)
    
    # Initialize PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        precision=16,  # Using mixed precision for efficiency
    )
    
    # Set up experiment manager for logging
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Initialize model with configuration
    model = nemo_duplex_s2s.models.SALM(OmegaConf.to_container(cfg.model, resolve=True))
    
    # Create dataset and datamodule
    dataset = SALMDataset(
        tokenizer=model.tokenizer,
        audio_locator_tag=model.audio_locator_tag
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    
    # Train the model
    trainer.fit(model, datamodule)
    
    # Save the trained model
    model.save_to("path/to/save/model.nemo")

Example Using Command-Line Training Script
------------------------------------------

Alternatively, you can train a model using the provided training scripts in the examples directory:

.. code-block:: bash

    # Train a SALM model
    python examples/duplex_s2s/salm_train.py \
      --config-path=conf \
      --config-name=salm

    # For inference/evaluation 
    python examples/duplex_s2s/salm_eval.py \
      pretrained_name=/path/to/checkpoint \
      inputs=/path/to/test_manifest \
      batch_size=64 \
      max_new_tokens=128 \
      output_manifest=generations.jsonl

For more detailed information on training at scale, model parallelism, and SLURM-based training, see :doc:`training and scaling <training_and_scaling>`.

Collection Structure
------------------

The duplex_s2s collection is organized into the following key components:

- **Models**: Contains implementations of DuplexS2SModel, DuplexS2SSpeechDecoderModel, and SALM
- **Modules**: Contains audio perception and speech generation modules
- **Data**: Includes dataset classes and data loading utilities

The collection is designed to be lightweight and flexible, allowing for easy integration with different LLMs while maintaining support for advanced features like model parallelism for large-scale training.

Duplex S2S Documentation
-----------------------

For more information, see additional sections in the Duplex S2S docs:

.. toctree::
   :maxdepth: 1

   models
   datasets
   configs
   training_and_scaling
