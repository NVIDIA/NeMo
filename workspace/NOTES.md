
# Speech SSL and SpeechLM


## Key Components

- Random-projection Quantization (RQ)
  - `nemo.collections.asr.modules.ssl_modules.quantizer.py`
- Multi-decoder and multi-softmax
  - `nemo.collections.asr.modules.ssl_modules.multi_softmax_decoder.py`
  - `nemo.collections.asr.losses.ssl_losses.mlm.MultiMLMLoss` 
- Random Block Masking (RBM)
  - `nemo.collections.asr.modules.ssl_modules.masking.RandomBlockMasking`
- Speech MLM Model with Masking-after-Convolutional (MAC) sub-sampling (similar to Wav2Vec-BERT)
  - `nemo.collections.asr.models.slm_models.SelfSupervisedConvMLMModel`
  - `./speech_pretrain_mac.py`
- Speech MLM Model with Masking log-mel spectrogram (BEST-RQ)
  - `nemo.collections.asr.models.slm_models.SelfSupervisedRandomQuantizationModel`
  - `./speech_pretrain.py`
