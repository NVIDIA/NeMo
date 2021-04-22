ASR Language Modeling
=====================

Language models have shown to help the accuracy of ASR models. NeMo support the following two approaches to incorporate language models into the ASR models:
    + N-gram Language Modelling
    + Neural Rescoring

It is possible to use both approaches on the same ASR model.


.. _ngram_modelling:

N-gram Language Modelling
-------------------------

In this approach, an N-gram LM is trained on text data, then it is used in fusion with beam search decoding to find the best candidates.
The beam search decoders in NeMo support language models trained by KenLM library `https://github.com/kpu/kenlm <https://github.com/kpu/kenlm>`__.
The beam search decoders and KenLM library are not installed by default in NeMo, and you need to install them to be
able to use beam search decoding and N-gram LM. Please refer to 'scripts/ngram_lm/install_beamsearch_decoders.sh'
on how to install them.

NeMo supports both character-based and BPE-based models for N-gram LMs. An N-gram LM can be used with beam search
decoders on top of the ASR models to produce more accurate candidates. The beam search decoder would incorporate
the scores produced by the N-gram LM into its score calculations as the following:

.. code::

    final_score = acoustic_score + beam_alpha*lm_score - beam_beta*seq_length

where acoustic_score is the score predicted by the acoustic encoder and lm_score is the one estimated by the LM.
Parameter 'beam_alpha' specifies amount of importance to place on the N-gram language model, and 'beam_beta' is a penalty term given to longer word sequences.

Train N-gram LM
---------------
The script to train an N-gram language model with KenLM can be found at
`scripts/asr_language_modelling/ngram_lm/train_kenlm.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/asr_language_modelling/ngram_lm/train_kenlm.py>`__.

This script would train an N-gram language model with KenLM library (https://github.com/kpu/kenlm) which can be used
with the beam search decoders on top of the ASR models. This script supports both character level and BPE level
encodings and models which is detected automatically from the type of the model.


You may train the N-gram model as the following:

.. code::

    python train_kenlm.py --nemo_model_file <path to the .nemo file of the model> \
                              --train_file <path to the training text or JSON manifest file \
                              --kenlm_bin_path <path to the bin folder of KenLM library> \
                              --kenlm_model_file <path to store the binary KenLM model> \
                              --ngram_length <order of N-gram model>

The train file specified by '--trainf_file' can be a text file or JSON manifest. If the file's extension is anything
other than '.json', it assumes that data format is plain text. For plain text format, each line should contain one
sample. For JSON manifest file, the file need to contain JSON formatted samples per each line. It extracts the 'text'
field of each line.

After the N-gram model is trained, it is stored at the path specified by '--kenlm_model_file'.

+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| **Parameter**    | **Data Type** | **Default** | **Description**                                                                                |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| nemo_model_file  | str           | Required    | The path of the '.nemo' file of the ASR model. It is needed to extract the tokenizer.          |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| train_file       | str           | Required    | Path to the training file, it can be a text file or JSON manifest.                             |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| kenlm_model_file | str           | Required    | The path to store the KenLM binary model file.                                                 |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| kenlm_bin_path   | str           | Required    | The path to the bin folder of KenLM. It is a folder named 'bin' under where KenLM is installed.|
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| ngram_length     | int           | Required    | Specifies order of N-gram LM.                                                                  |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+
| do_lower_case    | bool          | ``False``   | Whether to share the tokenizer between the encoder and decoder.                                |
+------------------+---------------+-------------+------------------------------------------------------------------------------------------------+

Recommend to use 6 as the order of the N-gram model for BPE-based models. Higher orders may need the re-compilation of KenLM to support it.


Evaluate by Beam Search Decoding and N-gram LM
----------------------------------------------

The script to evaluate an ASR model with N-gram models can be found at
`scripts/asr_language_modelling/ngram_lm/eval_beamsearch_ngram.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/asr_language_modelling/ngram_lm/eval_beamsearch_ngram.py>`__.
It can evaluate a model in three modes of 'greedy', 'beamsearch', and 'beamsearch_ngram' by setting the argument '--decoding_mode'.
The mode of 'beamsearch' would evaluate by beam search decoding without any language model.

You may evaluate an ASR model as the following:

.. code::

    python eval_beamsearch_ngram.py --nemo_model_file <path to the .nemo file of the model> \
                                         --input_manifest <path to the evaluation JSON manifest file \
                                         --kenlm_model_file <path to the binary KenLM model> \
                                         --acoustic_batch_size <batch size for calculating log probabilities> \
                                         --beam_width <list of the beam widths> \
                                         --beam_alpha <list of the beam alphas> \
                                         --beam_width <list of the beam betas> \
                                         --preds_output_folder <optional folder to store the predictions>
                                         --decoding_mode beam_search_ngram

It would report the performances in terms of Word Error Rate (WER) and Character Error Rate (CER).
You may use '--use_amp' to speed up the calculation of log probabilities.
You may find more detail on the arguments and how to use it inside the script.

There is also a tutorial to learn more about evaluating the ASR models with N-gram LM here:
`Offline ASR Inference with Beam Search and External Language Model Rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/asr/Offline_ASR.ipynb>`_

Hyperparameter Grid Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

Beam search decoding with N-gram LM has three main hyperparameters: 'beam_width', 'beam_alpha', and 'beam_beta'.
The accuracy of the model is dependent to the values of these parameters, specially beam_alpha and beam_beta.
You may specify a single or list of values for each of these parameters to perform grid search. It would perform the
beam search decoding on all the combinations of the these three hyperparameters.
For instance, the following set of parameters would results in 2*1*2=4 beam search decodings:

.. code::
    python eval_beamsearch_ngram.py ... \
                        --beam_width 64 128 \
                        --beam_alpha 1.0 \
                        --beam_beta 1.0 0.5



.. _neural_rescoring:
Neural Rescoring
----------------

In this approach a neural network is used which can gives scores to a candidate. A candidate is the text transcript predicted by the decoder of the ASR model.
The top K candidates produced by the beam search decoding (beam width of K) are given to a neural language model to rank them.
Ranking can be done by a language model which gives a score to each candidate.
This score is usually combined with the scores from the beam search decoding to produce the final scores and rankings.
An example script to train such a language model with Transformer can be found at `examples/nlp/language_modelling/transformer_lm.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modelling/transformer_lm.py>`__.
