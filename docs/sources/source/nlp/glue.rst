
Tutorial
========

In this tutorial, we are going to describe how to finetune a BERT-like models based on `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ :cite:`nlp-qa-devlin2018bert` on GLUE tasks. 
All code used in this tutorial is based on ``examples/nlp/glue_benchmark/glue_benchmark_with_bert.py``.

GLUE tasks description
----------------------

`GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding <https://openreview.net/pdf?id=rJ4km2R5t7>`_ :cite:`nlp-glue-wang2018glue` includes 9 natural language understanding tasks. 
Note, it's recommended to finetune you model on each task separately.

Single-Sentence Tasks:
**CoLA** The Corpus of Linguistic Acceptability :cite:`nlp-glue-warstadt2018neural` is a set of English sentences labeled as grammatical or ungrammatical from published linguistics literature.
Evaluation metric: Matthew's Corrrelation.
SST-2 The Stanford Sentiment Treebank :cite:`nlp-glue-socher2013recursive` consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given
sentence: positive or negative.  Evaluation metric: accuracy.

Similarity and Paraphrase tasks:
- MRPC The Microsoft Research Paraphrase Corpus :cite:`nlp-glue-dolan-brockett-2005-automatically` is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
Evaluation metrins: accuracy and F1 score.
- `QQP The Quora Question Pairs2 <https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs>` dataset is a collection of question pairs from the community question-answering website Quora.
The task is to determine whether a pair of questions are semantically equivalent. Evaluation metrics: accuracy and F1 score. 
- STS-B The Semantic Textual Similarity Benchmark :cite:`nlp-glue-cer2017semeval` is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.

Inference Tasks:
- MNLI The Multi-Genre Natural Language Inference Corpus :cite:`nlp-glue-williams2017broad` is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence
and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). 
The task has the matched (in-domain) and mismatched (cross-domain) sections, evaluation metric: accuracy.
 
We also use and recommend the SNLI corpus (Bowman et al., 2015) as 550k examples of auxiliary training data.
QNLI The Stanford Question Answering Dataset (Rajpurkar et al. 2016) is a question-answering
dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn
from Wikipedia) contains the answer to the corresponding question (written by an annotator). We
convert the task into sentence pair classification by forming a pair between each question and each
sentence in the corresponding context, and filtering out pairs with low lexical overlap between the
question and the context sentence. The task is to determine whether the context sentence contains
the answer to the question. This modified version of the original task removes the requirement that
the model select the exact answer, but also removes the simplifying assumptions that the answer
is always present in the input and that lexical overlap is a reliable cue. This process of recasting
existing datasets into NLI is similar to methods introduced in White et al. (2017) and expanded
upon in Demszky et al. (2018). We call the converted dataset QNLI (Question-answering NLI).3
RTE The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim
et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are
constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where
for three-class datasets we collapse neutral and contradiction into not entailment, for consistency.
WNLI The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task
in which a system must read a sentence with a pronoun and select the referent of that pronoun from
a list of choices. The examples are manually constructed to foil simple statistical methods: Each
one is contingent on contextual information provided by a single word or phrase in the sentence





`GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding <https://openreview.net/pdf?id=rJ4km2R5t7>`_ :cite:`nlp-glue-wang2018glue`.

Currently, there are 4 pretrained back-bone models supported, on which the question answering task SQuAD can be fine-tuned:
Megatron-LM BERT, BERT, ALBERT and RoBERTa. These are pretrained model checkpoints from `transformers <https://huggingface.co/transformers>`__ . Apart from these, the user can also do fine-tuning
on a custom BERT checkpoint, specified by the `--bert_checkpoint` argument.
The pretrained back-bone models can be specified `--pretrained_model_name`.
See the list of available pre-trained models
`here <https://huggingface.co/transformers/pretrained_models.html>`__. 

.. _pretrained_models_squad:

Download pretrained models
--------------------------

Finetuned SQuAD models and model configuration files can be downloaded at following links.

BERT Base uncased models (~330M parameters) finetuned on SQuADv1.1 or SQuADv2.0 dataset: 
`https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedsquadv1 <https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedsquadv1>`__
`https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedsquadv2 <https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedsquadv2>`__

BERT Large uncased models (~110M parameters) finetuned on SQuADv1.1 or SQuADv2.0 dataset: 
`https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedsquadv1 <https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedsquadv1>`__
`https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedsquadv2 <https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedsquadv2>`__


.. tip::

    For pretraining BERT in NeMo and pretrained model checkpoints go to `BERT pretraining <https://nvidia.github.io/NeMo/nlp/bert_pretraining.html>`__.

Model results:
(dev set, single model, single-task finetuning)

+---------------------------------------------+--------+--------+--------+--------+
|                                             | SQuADv1.1       | SQuADv2.0       |
+                                             +--------+--------+--------+--------+
|  Model                                      | EM     |  F1    |  EM    |  F1    |
+=============================================+========+========+========+========+
| BERT-base-uncased                           | 82.74% | 89.79% | 71.24% | 74.32% |
+---------------------------------------------+--------+--------+--------+--------+
| BERT-large-uncased                          | 85.79% | 92.28% | 80.17% | 83.32% |
+---------------------------------------------+--------+--------+--------+--------+

On a DGX1 with 8 V100 16GB training on SQuADv1.1 with the default script parameters takes between 14-18 minutes.

Preliminaries
-------------

**Model details**
This model trains token-level classifier to predict the start and end position of the answer span in context.
The loss is composed of the sum of the cross entropy loss of the start `S_loss` and end positions `E_loss`:

        `S_loss` + `E_loss`

At inference, the longest answer span that minimizes this loss is used as prediction.

**Datasets.** 

This model can work with any dataset that follows the format:

    * training file: a `json` file of this structure

    {"data":[{"title": "string", "paragraphs": [{"context": "string", "qas": [{"question": "string", "is_impossible": "bool", "id": "number", "answers": [{"answer_start": "number", "text": "string", }]}]}]}]}
    "answers" can also be empty if the model should also learn questions with impossible answers. In this case pass `--version_2_with_negative`

    * evaluation file: a `json` file that follows the training file format
      only that it can provide more than one entry for "answers" to the same question

    * test file: a `json` file that follows the training file format
      only that it does not require the "answers" keyword. 

Currently, the datasets that we provide pre-processing script for is SQuAD v1.1 and v2.0 
which can be downloaded
from `https://rajpurkar.github.io/SQuAD-explorer/ <https://rajpurkar.github.io/SQuAD-explorer/>`_.
You can find the pre-processing script in ``examples/nlp/question_answering/get_squad.py``.


Code structure
--------------

First, we instantiate Neural Module Factory which defines 1) backend (PyTorch), 2) mixed precision optimization level,
3) local rank of the GPU, and 4) an experiment manager that creates a timestamped folder to store checkpoints, relevant outputs, log files, and TensorBoard graphs.

    .. code-block:: python
    
        import nemo
        import nemo.collections.nlp as nemo_nlp
        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                               local_rank=args.local_rank,
                                               optimization_level=args.amp_opt_level,
                                               log_dir=work_dir,
                                               create_tb_writer=True,
                                               files_to_copy=[__file__],
                                               add_time_to_log_dir=True)

Next, we define all Neural Modules participating in our question answering classification pipeline.

    * Process data: the `BertQuestionAnsweringDataLayer` is supposed to do the preprocessing of raw data into the format data supported by `SquadDataset`.
    
    Training and evaluation each require their own `BertQuestionAnsweringDataLayer`. 
    DataLayer is an extra layer to do the semantic checking for your dataset and convert it into DataLayerNM. 

    .. code-block:: python

        data_layer = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
                                mode="train",
                                data_file=args.train_file,
                                tokenizer=tokenizer,
                                batch_size=args.batch_size,
                                version_2_with_negative=args.version_2_with_negative,
                                max_query_length=args.max_query_length,
                                max_seq_length=args.max_seq_length,
                                doc_stride=args.doc_stride,
                                use_cache=args.use_data_cache)

        
        data_layer_eval = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
                                mode='eval',
                                data_file=args.eval_file,
                                tokenizer=tokenizer,
                                batch_size=args.batch_size,
                                version_2_with_negative=args.version_2_with_negative,
                                max_query_length=args.max_query_length,
                                max_seq_length=args.max_seq_length,
                                doc_stride=args.doc_stride,
                                use_cache=args.use_data_cache)

    * Load the pretrained model and get the hidden states for the corresponding inputs.

    .. code-block:: python
        
        args.pretrained_model_name = "bert-base-uncased"
        model = nemo_nlp.nm.trainables.huggingface.BERT(args.pretrained_model_name)
        # or for RoBERTa
        args.pretrained_model_name = "roberta-base"
        model = nemo_nlp.nm.trainables.huggingface.Roberta(args.pretrained_model_name)
        # or for Albert
        args.pretrained_model_name = "albert-base-v1"
        model = nemo_nlp.nm.trainables.huggingface.Albert(args.pretrained_model_name)

    * Define the tokenizer which transforms text into BERT tokens, using `NemoBertTokenizer`. This will tokenize text following the mapping of the original BERT model.

    .. code-block:: python

        hidden_size = model.hidden_size
        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)


    * Create the classifier head for our task.

    .. code-block:: python

        qa_head = nemo_nlp.nm.trainables.TokenClassifier(
                                hidden_size=hidden_size,
                                num_classes=2,
                                num_layers=1,
                                log_softmax=False)

    * Create loss function

    .. code-block:: python

        loss_fn = nemo_nlp.nm.losses.SpanningLoss()

    * Create the pipelines for the train and evaluation processes. 

    .. code-block:: python

        # training graph
        input_data = data_layer()
        hidden_states = model(input_ids=input_data.input_ids,
                        token_type_ids=input_data.input_type_ids,
                        attention_mask=input_data.input_mask)

        qa_logits = qa_head(hidden_states=hidden_states)
        loss_outputs = squad_loss(
            logits=qa_logits,
            start_positions=input_data.start_positions,
            end_positions=input_data.end_positions)
        train_tensors = [loss_outputs.loss]

        # evaluation graph
        input_data_eval = data_layer_eval()

        hidden_states_eval = model(
            input_ids=input_data_eval.input_ids,
            token_type_ids=input_data_eval.input_type_ids,
            attention_mask=input_data_eval.input_mask)

        qa_logits_eval = qa_head(hidden_states=hidden_states_eval)
        loss_outputs_eval = squad_loss(
            logits=qa_logits_eval,
            start_positions=input_data_eval.start_positions,
            end_positions=input_data_eval.end_positions)
        eval_tensors = [input_data_eval.unique_ids, loss_outputs_eval.start_logits, loss_outputs_eval.end_logits]



    * Create relevant callbacks for saving checkpoints, printing training progresses and evaluating results.

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=args.step_freq,
            tb_writer=neural_factory.tb_writer)


        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x:
                eval_epochs_done_callback(
                    x, eval_data_layer=data_layer_eval,
                    do_lower_case=args.do_lower_case,
                    n_best_size=args.n_best_size,
                    max_answer_length=args.max_answer_length,
                    version_2_with_negative=args.version_2_with_negative,
                    null_score_diff_threshold=args.null_score_diff_threshold),
                tb_writer=neural_factory.tb_writer,
                eval_step=args.eval_step_freq)

        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=args.save_epoch_freq,
            step_freq=args.save_step_freq)

    * Finally, we define the optimization parameters and run the whole pipeline.

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)

        nf.train(tensors_to_optimize=train_tensors,
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 lr_policy=lr_policy_fn,
                 optimizer=args.optimizer_kind,
                 optimization_params={"num_epochs": args.num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay})

Model training
--------------

To run on a single GPU, run:
    
    .. code-block:: python

        python question_answering_squad.py \
            ...
            
To train a question answering model on SQuAD using multi-gpu, run ``question_answering_squad.py`` located at ``examples/nlp/question_answering``:

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=8 question_answering_squad.py 
            --train_file <path to train file in *.json format>
            --eval_file <path to evaluation file in *.json format>
            --num_gpus 8
            --work_dir <where you want to log your experiment> 
            --amp_opt_level <amp optimization level> 
            --pretrained_model_name <type of model to use> 
            --bert_checkpoint <pretrained bert checkpoint>
            --bert_config <model configuration file>
            --mode "train_eval"
            ...

For model configuration files and checkpoints, see :ref:`pretrained_models_squad`.

To run evaluation:

    .. code-block:: python

        python question_answering_squad.py 
            --eval_file <path to evaluation file in *.json format>
            --checkpoint_dir <path to trained SQuAD checkpoint folder>
            --mode "eval"
            --output_prediction_file <path to output file where predictions are written into>
            ...

To run inference:

    .. code-block:: python

        python question_answering_squad.py 
            --test_file <path to evaluation file in *.json format>
            --checkpoint_dir <path to trained SQuAD checkpoint folder>
            --mode "test"
            --output_prediction_file <path to output file where predictions are written into>
            ...


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-GLUE
    :keyprefix: nlp-glue-