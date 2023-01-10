ASR evaluator
--------------------

A tool for thoroughly evaluating the performance of ASR models and other features such as Voice Activity Detection. 

Features:
   - Simple step to evaluate a model in all three modes currently supported by NeMo: offline, chunked, and offline_by_chunked.
   - On-the-fly data augmentation (such as silence, noise, etc.,) for ASR robustness evaluation. 
   - Investigate the model's performance by detailed insertion, deletion, and substitution error rates for each and all samples.
   - Evaluate models' reliability on different target groups such as gender, and audio length if metadata is presented.


ASR evaluator contains two main parts: 
- **ENGINE**. To conduct ASR inference.
- **ANALYST**. To evaluate model performance based on predictions. 

In Analyst, you can evaluate on all metadata if it presents in manifest. For exmaple, you can evaluate the peformance of model based on duration of each sample, such as how's the model peforms on samples smaller than 5s and longer than 5s by [[0,5][5,100000]] and get wer/cer of each slot. Or how's the model performs on postive (happy, laugh) or neural (neural) or negative mood (sad) as below. And if you set save_wer_per_class=True, it will calculate wer for all (i.e. above 5 classes + cry) classes presented in the data. 

```
analyst:   
   metadata:
        duration: 
            enable: True
            slot: [[0,2],[2,5],[5,10],[10,20],[20,100000]] 
            save_wer_per_class: False # whether to save wer for each presented class.

        emotion: 
            enable: True
            slot: [['happy','laugh'],['neural'],['sad']] # we could have 'cry' in data but not in slot we focus on.
            save_wer_per_class: False
 ```           
            
            
Check `./conf/eval.yaml` for the supported configuration. 

If you plan to evaluate/add new tasks such as Punctuation and Capitalization, add it to the engine.

Run
```
python asr_evaluator.py \
engine.pretrained_name="stt_en_conformer_transducer_large" \
engine.inference_mode.mode="offline" \
engine.test_ds.augmentor.noise.manifest_path=<manifest file for noise data>
```