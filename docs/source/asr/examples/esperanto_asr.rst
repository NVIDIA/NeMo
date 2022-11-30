########################################################################
Example: Esperanto ASR using Mozilla Common Voice Dataset
########################################################################

Training an ASR model for a new language can be a challenging task, because there are many specific features which may differ depending on the language characteristics and amount of training data. At the moment NeMo already has a detailed example (link) for Kinyarwanda ASR training. You can find all the information for getting the ASR model there (read it first). The aim of the current example is to describe the ASR model training for Esperanto language and show some useful practices that can improve recognition accuracy. 

The example covers next steps:
* Data preparation.
* Tokenization.
* Analysis of training parameters. 
* Training from scratch and finetuning.
* Model evaluation. 

**************************
Data preparation.
**************************
Mozilla Common Voice provides a dataset for Esperanto language with about 1400 hours of validated data (general details of data corpuses creation can be found `here <https://arxiv.org/abs/1912.0667>`). However, the final training dataset consists only of 250 hours because the next rules – “The train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 
