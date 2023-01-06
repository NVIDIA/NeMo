ASR evaluator
--------------------

A tool for thoroughly evaluating the performance of ASR models and other features such as Voice Activity Detection. 

Features:
   - Simple step to evaluate a model in all three modes currently supported by NeMo: offline, chunked, and offline_by_chunked.
   - On-the-fly data augmentation (silence, noise, etc.,) for ASR robustness evaluation. 
   - Investigate the model's performance by detailed insertion, deletion, and substitution error rates for each and all samples.
   - Evaluate models' reliability on different target groups such as gender, and audio length if metadata is presented.


ASR evaluator contains two main parts: 
- **ENGINE**. To conduct ASR inference.
- **ANALYST**. To evaluate model performance based on predictions. 

Check `./conf/eval.yaml` for the supported configuration. 

If you plan to evaluate/add new tasks such as Punctuation and Capitalization, add it to the engine.

Run
```
python asr_evaluator.py \
engine.pretrained_name="stt_en_conformer_transducer_large" \
engine.inference_mode.mode="offline" \
engine.test_ds.augmentor.noise.manifest_path=<manifest file for noise data>
```