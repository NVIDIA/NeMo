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

1. Download and preprocess Yago corpus
   `dataset_preparation/preprocess_yago.sh`

2. Feed Yago data to G2P to get phonemes.
   `dataset_preparation/run_g2p.sh`

3. Feed phonetic inputs to TTS, generate wav files  (Jocelyn's code)
   Feed wav files to ASR, get ASR hypotheses (mostly misspelled)
   `dataset_preparation/run_tts_and_asr.sh`

4. Align parallel corpus of "correct + misspelled" phrases with Giza++, count frequencies of ngram mappings.
   `dataset_preparation/get_ngram_mappings.sh`

5. Generate training data in the following way.
   Substitute Yago misspelled phrases into longer sentences to get synthetic data.
   Given a set of custom phrases, index them with all possible "misspelled" ngrams.
   Then given an ASR hypothesis look all its ngrams in the index and select top-10 phrases with largest coverage.
   `dataset_preparation/build_training_data.sh`
 
6. Train spellchecking model.
   `run_training.sh`

7. Prepare Spoken Wikipedia corpus.
   Use
   [NEMO_ROOT]/scripts/dataset_processing/spoken_wikipedia/run.sh
   but replace in that bash-script
   ```
      mkdir ${INPUT_DIR}_prepared/audio
      mkdir ${INPUT_DIR}_prepared/text
      python ${NEMO_PATH}/scripts/dataset_processing/spoken_wikipedia/preprocess.py --input_folder ${INPUT_DIR} --destination_folder ${INPUT_DIR}_prepared
   ```

   with
   ```
      mkdir ${INPUT_DIR}_prepared/audio
      mkdir ${INPUT_DIR}_prepared/text
      mkdir ${INPUT_DIR}_prepared/vocabs
      python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/preprocess_spoken_wikipedia_and_create_vocabs.py --input_folder ${INPUT_DIR} --destination_folder ${INPUT_DIR}_prepared
   ```
   It does the same preprocessing but also creates vocabulary files to be used later.

8. Test on Spoken Wikipedia.
   `evaluation/test_on_spoken_wikipedia.sh`

   The evaluation folder structure should look like this:
   ```
   ├── english
   |   |── ...
   │   ├── Zinc
   │   └── Zorbing
   ├── english_prepared
   │   ├── audio
   │   ├── text
   │   └── vocabs
   ├── english_result
   │   ├── clips
   │   ├── hypotheses
   │   ├── logs
   │   ├── manifests
   │   ├── processed
   │   │   └── en_grammars
   │   ├── segments
   │   ├── spellchecker_input
   │   ├── spellchecker_output
   │   └── verified_segments
   ├── replacement_vocab_filt.txt [file generated during dataset_preparation/get_ngram_mappings.sh]
   └── test_on_spoken_wikipedia.sh
   ```

