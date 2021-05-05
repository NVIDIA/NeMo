#####################
ASR Language Modeling
#####################

Language models have shown to help the accuracy of ASR models. NeMo support the following two approaches to incorporate language models into the ASR models:

* :ref:`ngram_modeling`
* :ref:`neural_rescoring`

It is possible to use both approaches on the same ASR model.


.. _ngram_modeling:

************************
N-gram Language Modeling
************************

In this approach, an N-gram LM is trained on text data, then it is used in fusion with beam search decoding to find the
best candidates. The beam search decoders in NeMo support language models trained with KenLM library (
`https://github.com/kpu/kenlm <https://github.com/kpu/kenlm>`__).
The beam search decoders and KenLM library are not installed by default in NeMo, and you need to install them to be
able to use beam search decoding and N-gram LM.
Please refer to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.

NeMo supports both character-based and BPE-based models for N-gram LMs. An N-gram LM can be used with beam search
decoders on top of the ASR models to produce more accurate candidates. The beam search decoder would incorporate
the scores produced by the N-gram LM into its score calculations as the following:

.. code::

    final_score = acoustic_score + beam_alpha*lm_score + beam_beta*seq_length

where acoustic_score is the score predicted by the acoustic encoder and lm_score is the one estimated by the LM.
Parameter 'beam_alpha' specifies amount of importance to place on the N-gram language model, and 'beam_beta' is a
penalty term to consider the sequence length in the scores. Larger alpha means more importance on the LM and less
importance on the acoustic model. Negative values for beta will give penalty to longer sequences and make the decoder
to prefer shorter predictions, while positive values would result in longer candidates.


Train N-gram LM
===============

The script to train an N-gram language model with KenLM can be found at
`scripts/asr_language_modeling/ngram_lm/train_kenlm.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/asr_language_modeling/ngram_lm/train_kenlm.py>`__.

This script would train an N-gram language model with KenLM library which can be used with the beam search decoders
on top of the ASR models. This script supports both character level and BPE level encodings and models which is
detected automatically from the type of the model.


You may train the N-gram model as the following:

.. code::

    python train_kenlm.py --nemo_model_file <path to the .nemo file of the model> \
                              --train_file <path to the training text or JSON manifest file \
                              --kenlm_bin_path <path to the bin folder of KenLM library> \
                              --kenlm_model_file <path to store the binary KenLM model> \
                              --ngram_length <order of N-gram model>

The train file specified by '--trainf_file' can be a text file or JSON manifest. If the file's extension is anything
other than '.json', it assumes that data format is plain text. For plain text format, each line should contain one
sample. For JSON manifest file, the file need to contain json formatted samples per each line like this:

.. code::

    {"audio_filepath": "/data_path/file1.wav", "text": "The transcript of the audio file."}

It just extracts the 'text' field from each line to create the training text file. After the N-gram model is trained,
it is stored at the path specified by '--kenlm_model_file'.

The following is the list of the arguments for the training script:

+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| **Argument**     | **Type** | **Default** | **Description**                                                                                |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| nemo_model_file  | str      | Required    | The path of the '.nemo' file of the ASR model. It is needed to extract the tokenizer.          |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| train_file       | str      | Required    | Path to the training file, it can be a text file or JSON manifest.                             |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| kenlm_model_file | str      | Required    | The path to store the KenLM binary model file.                                                 |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| kenlm_bin_path   | str      | Required    | The path to the bin folder of KenLM. It is a folder named 'bin' under where KenLM is installed.|
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| ngram_length**   | int      | Required    | Specifies order of N-gram LM.                                                                  |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+
| do_lower_case    | bool     | ``False``   | Whether to make the training text all lower case.                                              |
+------------------+----------+-------------+------------------------------------------------------------------------------------------------+

** Note: Recommend to use 6 as the order of the N-gram model for BPE-based models. Higher orders may need the re-compilation of KenLM to support it.

Evaluate by Beam Search Decoding and N-gram LM
==============================================

NeMo's beam search decoders are capable of using the KenLM's N-gram models to find the best candidates.
The script to evaluate an ASR model with beam search decoding and N-gram models can be found at
`scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py>`__.

You may evaluate an ASR model as the following:

.. code::

    python eval_beamsearch_ngram.py --nemo_model_file <path to the .nemo file of the model> \
                                         --input_manifest <path to the evaluation JSON manifest file \
                                         --kenlm_model_file <path to the binary KenLM model> \
                                         --acoustic_batch_size <batch size for calculating log probabilities> \
                                         --beam_width <list of the beam widths> \
                                         --beam_alpha <list of the beam alphas> \
                                         --beam_width <list of the beam betas> \
                                         --preds_output_folder <optional folder to store the predictions> \
                                         --decoding_mode beam_search_ngram

It can evaluate a model in the three following modes by setting the argument '--decoding_mode':

* greedy: Just greedy decoding is done, and no beam search decoding is performed.
* beamsearch: The beam search decoding is done but without using the N-gram language model, final results would be equivalent to setting the weight of LM (beam_beta) to zero.
* beamsearch_ngram: The beam search decoding is done with N-gram LM.


The 'beamsearch' mode would evaluate by beam search decoding without any language model.
It would report the performances in terms of Word Error Rate (WER) and Character Error Rate (CER). Moreover,
the WER/CER of the model when the best candidate is selected among the candidates is also reported as the best WER/CER.
It can be an indicator of how good the predicted candidates are.

The script would initially load the ASR model and predict the outputs of the model's encoder as log probabilities.
This part would be computed in batches on a device selected by '--device', which can be CPU ('--device='cpu') or a
single GPU ('--device=cuda:0'). The batch size of this part can get specified by '--acoustic_batch_size'. You may use
the largest batch size feasible to speed up the step of calculating the log probabilities. You may also use '--use_amp'
to speed up the calculation of log probabilities and make it possible to use larger sizes for '--acoustic_batch_size'.
Currently multi-GPU is not supported for calculating the log probabilities, but using '--probs_cache_file' can help.
It stores the log probabilities produced from the model's encoder into a pickle file so that next time the first step
can get skipped.

The following is the list of the arguments for the evaluation script:

+---------------------+--------+------------------+-------------------------------------------------------------------------+
| **Argument**        |**Type**| **Default**      | **Description**                                                         |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| nemo_model_file     | str    | Required         | The path of the '.nemo' file of the ASR model to extract the tokenizer. |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| input_manifest      | str    | Required         | Path to the training file, it can be a text file or JSON manifest.      |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| kenlm_model_file    | str    | Required         | The path to store the KenLM binary model file.                          |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| preds_output_folder | str    | None             | The path to an optional folder to store the predictions.                |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| probs_cache_file    | str    | None             | The cache file for storing the outputs of the model.                    |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| acoustic_batch_size | int    | 16               | The batch size to calculate log probabilities.                          |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| use_amp             | bool   | ``False``        | Whether to use AMP if available to calculate log probabilities.         |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| device              | str    | cuda             | The device to load the model onto to calculate log probabilities.       |
|                     |        |                  | It can 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...                           |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| decoding_mode       | str    | beamsearch_ngram | The decoding scheme to be used for evaluation.                          |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beam_width          | float  | Required         | The width or list of the widths of the beam search decoding.            |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beam_alpha          | float  | Required         | The alpha parameter or list of the alphas for the beam search decoding. |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beam_beta           | float  | Required         | The beta parameter or list of the betas for the beam search decoding.   |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beam_batch_size     | int    | 128              | The batch size to be used for beam search decoding.                     |
|                     |        |                  | Larger batch size can be a little faster, but uses larger memory.       |
+---------------------+--------+------------------+-------------------------------------------------------------------------+

Width of the beam search ('--beam_width') specifies the number of top candidates/predictions the beam search decoder
would search for. Larger beams result in more accurate but slower predictions.

There is also a tutorial to learn more about evaluating the ASR models with N-gram LM here:
`Offline ASR Inference with Beam Search and External Language Model Rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.0/tutorials/asr/Offline_ASR.ipynb>`_

Hyperparameter Grid Search
--------------------------

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

****************
Neural Rescoring
****************

In this approach a neural network is used which can gives scores to a candidate. A candidate is the text transcript predicted by the decoder of the ASR model.
The top K candidates produced by the beam search decoding (beam width of K) are given to a neural language model to rank them.
Ranking can be done by a language model which gives a score to each candidate.
This score is usually combined with the scores from the beam search decoding to produce the final scores and rankings.
An example script to train such a language model with Transformer can be found at `examples/nlp/language_modeling/transformer_lm.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/transformer_lm.py>`__.
