# Text-to-Speech (TTS) Examples

This directory contains example scripts for various Text-to-Speech models and tasks in NeMo.

## Available Models

### FastPitch
- `fastpitch.py` - Basic FastPitch model training
- `fastpitch_finetune.py` - Fine-tuning FastPitch on custom data
- `fastpitch_finetune_adapters.py` - Fine-tuning FastPitch using adapter-based PEFT
- `fastpitch_ssl.py` - FastPitch with self-supervised learning

### HiFi-GAN
- `hifigan.py` - HiFi-GAN vocoder training
- `hifigan_finetune.py` - Fine-tuning HiFi-GAN on custom data

### Other Models
- `tacotron2.py` - Tacotron2 model training
- `tacotron2_finetune.py` - Fine-tuning Tacotron2
- `vits.py` - VITS model training
- `waveglow.py` - WaveGlow vocoder training
- `univnet.py` - UniNet vocoder training
- `radtts.py` - RAD-TTS model training
- `magpietts.py` - Magpie TTS model training
- `mixer_tts.py` - Mixer-TTS model training

### Utility Scripts
- `test_tts_infer.py` - Test inference with various TTS models
- `aligner.py` - Text alignment tool
- `aligner_heteronym_disambiguation.py` - Heteronym disambiguation for alignment
- `spectrogram_enhancer.py` - Enhance spectrograms for better quality
- `audio_codec.py` - Audio codec utilities
- `ssl_tts.py` - Self-supervised learning for TTS

## Usage Examples

### Training FastPitch
```bash
# Basic training
python fastpitch.py \
    train_dataset=<path_to_train_manifest> \
    validation_datasets=<path_to_val_manifest> \
    trainer.max_epochs=100 \
    trainer.accelerator=gpu \
    trainer.devices=1

# Fine-tuning
python fastpitch_finetune.py \
    model_path=<path_to_pretrained_model> \
    train_dataset=<path_to_train_manifest> \
    validation_datasets=<path_to_val_manifest> \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=1
```

### Training HiFi-GAN
```bash
# Basic training
python hifigan.py \
    train_dataset=<path_to_train_manifest> \
    validation_datasets=<path_to_val_manifest> \
    trainer.max_epochs=100 \
    trainer.accelerator=gpu \
    trainer.devices=1

# Fine-tuning
python hifigan_finetune.py \
    model_path=<path_to_pretrained_model> \
    train_dataset=<path_to_train_manifest> \
    validation_datasets=<path_to_val_manifest> \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=1
```

### Testing Inference
```bash
python test_tts_infer.py \
    model_path=<path_to_model> \
    text="Hello, this is a test." \
    output_path=output.wav \
    speaker_id=0  # Optional, for multi-speaker models
```

## Configuration

All scripts use Hydra for configuration management. You can find the default configurations in the `conf/` directory. To override any configuration parameter, add it to the command line:

```bash
python fastpitch.py \
    model.encoder.hidden_channels=256 \
    model.encoder.filter_channels=1024 \
    trainer.max_epochs=200
```

## Data Format

The training scripts expect data in the NeMo manifest format. Each line in the manifest should be a JSON object with the following fields:
```json
{
    "audio_filepath": "path/to/audio.wav",
    "text": "The text to be synthesized",
    "duration": 3.5,
    "speaker_id": 0  // Optional, for multi-speaker models
}
```

## Common Parameters

- `train_dataset`: Path to training manifest file
- `validation_datasets`: Path to validation manifest file
- `model_path`: Path to pretrained model for fine-tuning
- `trainer.max_epochs`: Number of training epochs
- `trainer.accelerator`: Training accelerator (gpu/cpu)
- `trainer.devices`: Number of devices to use
- `trainer.precision`: Training precision (16/32)

## Notes

1. For multi-GPU training, set `trainer.devices` to the number of GPUs you want to use
2. For mixed precision training, add `trainer.precision=16`
3. For distributed training, add `trainer.strategy=ddp`
4. Most models support both single-speaker and multi-speaker training
5. The scripts automatically handle data loading, preprocessing, and training loop

## Troubleshooting

1. If you encounter CUDA out of memory errors:
   - Reduce batch size
   - Use gradient accumulation
   - Use mixed precision training

2. If training is slow:
   - Enable mixed precision training
   - Use multiple GPUs
   - Optimize data loading with num_workers

3. If audio quality is poor:
   - Check audio preprocessing parameters
   - Verify data quality
   - Adjust model hyperparameters 