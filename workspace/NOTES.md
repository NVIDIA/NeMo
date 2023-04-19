
# Speech SSL and SpeechLM

## Plan
- [ ] Reproduce BEST-RQ results with Conformer and FastConformer
- [ ] Reproduce/match GoogleUSM results with semi-supervised training (audio-only, text-only, paired audio-text)
  - without using text-injection in GoogleUSM


## Datasets
- LibriLight (LL)
- Librispeech (LS)
- Multi-lingual LibriSpeech (MLS)
- Voxpopuli (VP)
- Mozilla Common Voice (MCV)

## Status
- [x] Implement Random-projection Quantization (RQ) with multiple decoders and softmax on different codebooks
  - `nemo.collections.asr.modules.ssl_modules.quantizer.py`
  - `nemo.collections.asr.modules.ssl_modules.multi_softmax_decoder.py`
- [x] Implement more precise random block masking mechanism
  - `nemo.collections.asr.losses.ssl_losses.mlm.MultiMLMLoss` 
- [ ] Train Conformer-L-RQ on LS data
- [ ] Train Conformer-L-RQ on LL data


