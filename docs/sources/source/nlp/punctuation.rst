Tutorial
========


An ASR system typically generates text with no punctuation and capitalization of the words. This tutorial explains how to implement a model in NeMo that will predict punctuation and capitalization for each word in a sentence to make ASR output more readable and to boost performance of the downstream tasks such as name entity recognition or machine translation. We'll show how to train network for this task using a pre-trained BERT model. 

.. tip::

    We recommend you to try this example in Jupyter notebook examples/nlp/token_classification/PunctuationWithBERT.ipynb.
    
    All code used in this tutorial is based on :ref:`punct_scripts`.
    For pretraining BERT in NeMo and pretrained model checkpoints go to `BERT pretraining <https://nvidia.github.io/NeMo/nlp/bert_pretraining.html>`__.


Task Description
----------------

For every word in our training dataset we're going to predict:

1. punctuation mark that should follow the word and
2. whether the word should be capitalized

In this model, we're jointly training 2 token-level classifiers on top of the pretrained BERT model: one classifier to predict punctuation and the other one - capitalization.

Dataset
-------

This model can work with any dataset as long as it follows the format specified below. For this tutorial, we're going to use the `Tatoeba collection of sentences`_. `This`_ script downloads and preprocesses the dataset. 

.. _Tatoeba collection of sentences: https://tatoeba.org/eng
.. _This: https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/token_classification/get_tatoeba_data.py


The training and evaluation data is divided into 2 files: text.txt and labels.txt. Each line of the text.txt file contains text sequences, where words are separated with spaces:
[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:

  ::
    
    when is the next flight to new york
    the next flight is ...
    ...

The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces.
Each label in labels.txt file consists of 2 symbols:

* the first symbol of the label indicates what punctuation mark should follow the word (where ``O`` means no punctuation needed);
* the second symbol determines if the word needs to be capitalized or not (where ``U`` indicates that the associated with this label word should be upper cased, and ``O`` - no capitalization needed.)

We're considering only commas, periods, and question marks for this task; the rest punctuation marks were removed.
Each line of the labels.txt should follow the format: 
[LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). For example, labels for the above text.txt file should be:

::
    
    OU OO OO OO OO OO OU ?U 
    OU OO OO OO ...
    ...

The complete list of all possible labels for this task is: ``OO``, ``,O``, ``.O``, ``?O``, ``OU``, ``,U``, ``.U``, ``?U``.

Code overview
-------------

First, let's set some parameters that we're going to need through out this tutorial:

    .. code-block:: python
        
        DATA_DIR = "PATH_TO_WHERE_THE_DATA_IS"
        WORK_DIR = "PATH_TO_WHERE_TO_STORE_CHECKPOINTS_AND_LOGS"
        PRETRAINED_BERT_MODEL = "bert-base-uncased"

        # model parameters
        BATCHES_PER_STEP = 1
        BATCH_SIZE = 128
        CLASSIFICATION_DROPOUT = 0.1
        MAX_SEQ_LENGTH = 64
        NUM_EPOCHS = 10
        LEARNING_RATE = 0.00002
        LR_WARMUP_PROPORTION = 0.1
        OPTIMIZER = "adam"
        STEP_FREQ = 200 # determines how often loss will be printed and checkpoint saved
        PUNCT_NUM_FC_LAYERS = 3
        NUM_SAMPLES = 100000

To download and preprocess a subset of the Tatoeba collection of sentences, run:

.. code-block:: bash
        
        python get_tatoeba_data.py --data_dir DATA_DIR --num_sample NUM_SAMPLES

Then, we need to create our neural factory with the supported backend. This tutorial assumes that you're training on a single GPU, with mixed precision (``optimization_level="O1"``). If you don't want to use mixed precision, set ``optimization_level`` to ``O0``.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=None,
                                           optimization_level="O1",
                                           log_dir=WORK_DIR,
                                           placement=nemo.core.DeviceType.GPU)

Next, we'll need to define our tokenizer and our BERT model. If you're using a standard BERT model, you should do it as follows. To see the full list of BERT model names, check out ``nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = nemo.collections.nlp.data.NemoBertTokenizer(pretrained_model=PRETRAINED_BERT_MODEL)
        bert_model = nemo_nlp.nm.trainables.huggingface.BERT(
            pretrained_model_name=PRETRAINED_BERT_MODEL)

Now, create the train and evaluation data layers:

    .. code-block:: python

        train_data_layer = nemo_nlp.nm.data_layers.PunctuationCapitalizationDataLayer(
                                            tokenizer=tokenizer,
                                            text_file=os.path.join(DATA_DIR, 'text_train.txt'),
                                            label_file=os.path.join(DATA_DIR, 'labels_train.txt'),
                                            max_seq_length=MAX_SEQ_LENGTH,
                                            batch_size=BATCH_SIZE)

        punct_label_ids = train_data_layer.dataset.punct_label_ids
        capit_label_ids = train_data_layer.dataset.capit_label_ids

        hidden_size = bert_model.hidden_size

        # Note that you need to specify punct_label_ids and capit_label_ids  - mapping form labels
        # to label_ids generated during creation of the train_data_layer to make sure that
        # the mapping is correct in case some of the labels from
        # the train set are missing in the dev set.
        eval_data_layer = nemo_nlp.BertPunctuationCapitalizationDataLayer(
                                            tokenizer=tokenizer,
                                            text_file=os.path.join(DATA_DIR, 'text_dev.txt'),
                                            label_file=os.path.join(DATA_DIR, 'labels_dev.txt'),
                                            max_seq_length=MAX_SEQ_LENGTH,
                                            batch_size=BATCH_SIZE,
                                            punct_label_ids=punct_label_ids,
                                            capit_label_ids=capit_label_ids)


Now, create punctuation and capitalization classifiers to sit on top of the pretrained BERT model and define the task loss function:

  .. code-block:: python

      punct_classifier = TokenClassifier(
                                         hidden_size=hidden_size,
                                         num_classes=len(punct_label_ids),
                                         dropout=CLASSIFICATION_DROPOUT,
                                         num_layers=PUNCT_NUM_FC_LAYERS,
                                         name='Punctuation')

      capit_classifier = TokenClassifier(hidden_size=hidden_size,
                                         num_classes=len(capit_label_ids),
                                         dropout=CLASSIFICATION_DROPOUT,
                                         name='Capitalization')


      # If you don't want to use weighted loss for Punctuation task, use class_weights=None
      punct_label_freqs = train_data_layer.dataset.punct_label_frequencies
      class_weights = nemo.collections.nlp.data.datasets.datasets_utils.calc_class_weights(punct_label_freqs)

      # define loss
      punct_loss = CrossEntropyLossNM(logits_dim=3, weight=class_weights)
      capit_loss = CrossEntropyLossNM(logits_dim=3)
      task_loss = LossAggregatorNM(num_inputs=2)


Below, we're passing the output of the datalayers through the pretrained BERT model and to the classifiers:

  .. code-block:: python

      input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = train_data_layer()

      hidden_states = bert_model(input_ids=input_ids,
                            token_type_ids=input_type_ids,
                            attention_mask=input_mask)

      punct_logits = punct_classifier(hidden_states=hidden_states)
      capit_logits = capit_classifier(hidden_states=hidden_states)

      punct_loss = punct_loss(logits=punct_logits,
                              labels=punct_labels,
                              loss_mask=loss_mask)
      capit_loss = capit_loss(logits=capit_logits,
                              labels=capit_labels,
                              loss_mask=loss_mask)
      task_loss = task_loss(loss_1=punct_loss,
                            loss_2=capit_loss)

      eval_input_ids, eval_input_type_ids, eval_input_mask, _, eval_subtokens_mask, eval_punct_labels, eval_capit_labels\
          = eval_data_layer()

      hidden_states = bert_model(input_ids=eval_input_ids,
                                 token_type_ids=eval_input_type_ids,
                                 attention_mask=eval_input_mask)

      eval_punct_logits = punct_classifier(hidden_states=hidden_states)
      eval_capit_logits = capit_classifier(hidden_states=hidden_states)



Now, we will set up our callbacks. We will use 3 callbacks:

* `SimpleLossLoggerCallback` prints loss values during training;
* `EvaluatorCallback` calculates the performance metrics for the dev dataset;
* `CheckpointCallback` is used to save and restore checkpoints.

    .. code-block:: python

        callback_train = nemo.core.SimpleLossLoggerCallback(
        tensors=[task_loss, punct_loss, capit_loss, punct_logits, capit_logits],
        print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
        step_freq=STEP_FREQ)

        train_data_size = len(train_data_layer)

        # If you're training on multiple GPUs, this should be
        # train_data_size / (batch_size * batches_per_step * num_gpus)
        steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

        # Callback to evaluate the model
        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_punct_logits,
                          eval_capit_logits,
                          eval_punct_labels,
                          eval_capit_labels,
                          eval_subtokens_mask],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x,
                                                                          punct_label_ids,
                                                                          capit_label_ids),
            eval_step=steps_per_epoch)

        # Callback to store checkpoints
        ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                                     step_freq=STEP_FREQ)

Finally, we'll define our learning rate policy and our optimizer, and start training:

    .. code-block:: python

        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[task_loss],
                 callbacks=[callback_train, callback_eval, ckpt_callback],
                 lr_policy=lr_policy,
                 batches_per_step=BATCHES_PER_STEP,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

Inference
---------

To see how the model performs, let's run inference on a few samples. We need to define a data layer for inference the same way we created data layers for training and evaluation.

.. code-block:: python

    queries = ['can i help you',
               'yes please',
               'we bought four shirts from the nvidia gear store in santa clara',
               'we bought four shirts one mug and ten thousand titan rtx graphics cards',
               'the more you buy the more you save']
    infer_data_layer = nemo_nlp.nm.data_layers.BertTokenClassificationInferDataLayer(
                                                            queries=queries,
                                                            tokenizer=tokenizer,
                                                            max_seq_length=MAX_SEQ_LENGTH,
                                                            batch_size=1)


Run inference, append punctuation and capitalize words based on the generated predictions:

.. code-block:: python

    input_ids, input_type_ids, input_mask, _, subtokens_mask = infer_data_layer()

    hidden_states = bert_model(input_ids=input_ids,
                                          token_type_ids=input_type_ids,
                                          attention_mask=input_mask)
    punct_logits = punct_classifier(hidden_states=hidden_states)
    capit_logits = capit_classifier(hidden_states=hidden_states)

    evaluated_tensors = nf.infer(tensors=[punct_logits, capit_logits, subtokens_mask],
                                 checkpoint_dir=WORK_DIR + '/checkpoints')



    # helper functions
    def concatenate(lists):
        return np.concatenate([t.cpu() for t in lists])

    punct_ids_to_labels = {punct_label_ids[k]: k for k in punct_label_ids}
    capit_ids_to_labels = {capit_label_ids[k]: k for k in capit_label_ids}

    punct_logits, capit_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]
    punct_preds = np.argmax(punct_logits, axis=2)
    capit_preds = np.argmax(capit_logits, axis=2)

    for i, query in enumerate(queries):
        logging.info(f'Query: {query}')

        punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
        capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
        words = query.strip().split()
        if len(punct_pred) != len(words) or len(capit_pred) != len(words):
            raise ValueError('Pred and words must be of the same length')

        output = ''
        for j, w in enumerate(words):
            punct_label = punct_ids_to_labels[punct_pred[j]]
            capit_label = capit_ids_to_labels[capit_pred[j]]

            if capit_label != 'O':
                w = w.capitalize()
            output += w
            if punct_label != 'O':
                output += punct_label
            output += ' '
        logging.info(f'Combined: {output.strip()}\n')

Inference results:
    
    ::

        Query: can i help you
        Combined: Can I help you?

        Query: yes please
        Combined: Yes, please.

        Query: we bought four shirts from the nvidia gear store in santa clara
        Combined: We bought four shirts from the Nvidia gear store in Santa Clara.

        Query: we bought four shirts one mug and ten thousand titan rtx graphics cards
        Combined: We bought four shirts, one mug, and ten thousand Titan Rtx graphics cards.

        Query: the more you buy the more you save
        Combined: The more you buy, the more you save.

.. _punct_scripts:

Training and inference scripts
------------------------------

To run the provided training script:

.. code-block:: bash

    python examples/nlp/token_classification/punctuation_capitalization.py --data_dir path_to_data --pretrained_bert_model=bert-base-uncased --work_dir path_to_output_dir

To run inference:

.. code-block:: bash

    python examples/nlp/token_classification/punctuation_capitalization_infer.py --punct_labels_dict path_to_data/punct_label_ids.csv --capit_labels_dict path_to_data/capit_label_ids.csv --work_dir path_to_output_dir/checkpoints/

Note, punct_label_ids.csv and capit_label_ids.csv files will be generated during training and stored in the data_dir folder.

Multi GPU Training
------------------

To run training on multiple GPUs, run

.. code-block:: bash

    export NUM_GPUS=2
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/nlp/token_classification/punctuation_capitalization.py --num_gpus $NUM_GPUS --data_dir path_to_data
