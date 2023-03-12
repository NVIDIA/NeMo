# Spellchecking model for ASR Customization

This model is inspired by Microsoft's paper https://arxiv.org/pdf/2203.00888.pdf, but does not repeat its implementation.
The goal is to build a model that gets as input a single ASR hypothesis (text) and a vocabulary of custom words/phrases and predicts which fragments in the ASR hypothesis should be replaced by which custom words/phrases if any.
Our model is non-autoregressive (NAR) based on transformer architecture (BERT with multiple separators).

As initial data we use about 5 mln entities from [YAGO corpus](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/). These entities are short phrases from Wikipedia headings.
In order to get misspelled predictions we feed these data to TTS model and then to ASR model.
Having a "parallel" corpus of "correct + misspelled" phrases, we use statistical machine translation techniques to create a dictionary of possible ngram mappings with their respective frequencies.
We create an auxiliary algorithm that takes as input a sentence (ASR hypothesis) and a large custom dictionary (e.g. 5000 phrases) and selects top 10 candidate phrases that are probably contained in this sentence in a misspelled way.
The task of our final neural model is to predict which fragments in the ASR hypothesis should be replaced by which of top-10 candidate phrases if any.
We use Spoken Wikipedia dataset for testing.

The pipeline consists of multiple steps:

0. [Optional] Train G2P model on cmudict
      See [google drive]/spellchecking_asr_customization/g2p
          run.sh(data preparation)
          train_g2p.sh(training),
          run_infer.sh(inference)

1. Download and preprocess Yago corpus and instruction on how to download full Wikipedia articles
   `dataset_preparation/preprocess_yago.sh`

2. Feed Yago data to G2P to get phonemes.
   `dataset_preparation/run_g2p.sh`

3. Feed phonetic inputs to TTS, generate wav files  (Jocelyn's code)
   Feed wav files to ASR, get ASR hypotheses (mostly misspelled)
   `dataset_preparation/run_tts_and_asr.sh`

4. Align parallel corpus of "correct + misspelled" phrases with Giza++, count frequencies of ngram mappings.
   `dataset_preparation/get_ngram_mappings.sh`

5. Generate training data. 
   `dataset_preparation/build_training_data.sh`
   Generate files config.json, label_map.txt, semiotic_classes.txt.
   `dataset_preparation/generate_configs.sh`
   [Optional] Convert training dataset to tarred files.
   `dataset_preparation/convert_dataset_to_tarred.sh`
 
6. Train spellchecking model.
   `run_training.sh`
   or 
   `run_training_tarred.sh`

7. Run evaluation.
   `evaluation/test_on_spoken_wikipedia.sh`
   or
   `evaluation/test_on_kensho.sh`
   or
   `evaluation/test_on_userlibri.sh`
