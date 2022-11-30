########################################################################
Example: Esperanto ASR using Mozilla Common Voice Dataset
########################################################################

Training an ASR model for a new language can be a challenging task, because there are many specific features which may differ depending on the language characteristics and amount of training data. At the moment NeMo already has a detailed example (link) for Kinyarwanda ASR training. You can find all the information for getting the ASR model there (read it first). The aim of the current example is to describe the ASR model training for Esperanto language and show some useful practices that can improve recognition accuracy. 

The example covers the next steps:

* Data preparation.
* Tokenization.
* Analysis of training parameters. 
* Training from scratch and finetuning.
* Model evaluation. 

**************************
Data preparation.
**************************
Mozilla Common Voice provides a dataset for Esperanto language with about 1400 hours of validated data (general details of data corpuses creation can be found `here <https://arxiv.org/abs/1912.0667>`_). However, the final training dataset consists only of 250 hours because the next rules – “The train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 

Download data:
#################################

To get data manifests for Esperanto you can use the default NeMo script `get_commonvoice_data.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/get_commonvoice_data.py>`_. But you need to change the variable COMMON_VOICE_URL to f"https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/{}/{}-{}.tar.gz".format(args.version, args.version, args.language)and add to json.dump parameter ensure_ascii=False (otherwise you will get problems with utf-8 symbols) and quoting=csv.QUOTE_NONE to csv.DictReader (unclosed quotes occurs sometimes in text). Whole modified script can be found here – link.

