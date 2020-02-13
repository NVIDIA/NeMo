Tutorial
========

Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this
tutorial. See the :ref:`installation` section for more details.

.. tip::

    For pretraining BERT in NeMo and pretrained model checkpoints go to `BERT pretraining <https://nvidia.github.io/NeMo/nlp/bert_pretraining.html>`__.



Introduction
------------

This tutorial explains how to implement named entity recognition (NER) in NeMo. We'll show how to do this with a pre-trained BERT model, or with one that you trained yourself! For more details, check out our BERT pretraining tutorial.

Download Dataset
----------------

`CoNLL-2003`_ is a standard evaluation dataset for NER, but any NER dataset will work. The only requirement is that the data is splitted into 2 files: text.txt and labels.txt. The text.txt files should be formatted like this:

.. _CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/

.. code-block::

    Jennifer is from New York City .
    She likes ...
    ...

The labels.txt files should be formatted like this:

.. code-block::

    B-PER O O B-LOC I-LOC I-LOC O
    O O ...
    ...

Each line of the text.txt file contains text sequences, where words are separated with spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces. Each line of the files should follow the format: [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).

You can use `this`_ to convert CoNLL-2003 dataset to the format required for training.


.. _this: https://github.com/NVIDIA/NeMo/blob/master/scripts/convert_iob_format_to_token_classification_format.py


Training
--------

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!
    See examples/nlp/NERWithBERT.ipynb

First, we need to create our neural factory with the supported backend. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision (``optimization_level="O0"``). If you want to use mixed precision, set ``optimization_level`` to ``O1`` or ``O2``.

    .. code-block:: python

        WORK_DIR = "output_ner"
        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=None,
                                           optimization_level="O0",
                                           log_dir=WORK_DIR,
                                           create_tb_writer=True)

Next, we'll need to define our tokenizer and our BERT model. There are a couple of different ways you can do this. Keep in mind that NER benefits from casing ("New York City" is easier to identify than "new york city"), so we recommend you use cased models.

If you're using a standard BERT model, you should do it as follows. To see the full list of BERT model names, check out ``nemo_nlp.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = NemoBertTokenizer(pretrained_model="bert-base-cased")
        bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-cased")

See examples/nlp/token_classification.py on how to use a BERT model that you pre-trained yourself.
Now, create the train and evaluation data layers:

    .. code-block:: python
    
        train_data_layer = nemo_nlp.BertTokenClassificationDataLayer(
            tokenizer=tokenizer,
            text_file=os.path.join(DATA_DIR, 'text_train.txt'),
            label_file=os.path.join(DATA_DIR, 'labels_train.txt'),
            max_seq_length=MAX_SEQ_LENGTH,
            batch_size=BATCH_SIZE)

        label_ids = train_data_layer.dataset.label_ids
        num_classes = len(label_ids)

        eval_data_layer = nemo_nlp.BertTokenClassificationDataLayer(
            tokenizer=tokenizer,
            text_file=os.path.join(DATA_DIR, 'text_dev.txt'),
            label_file=os.path.join(DATA_DIR, 'labels_dev.txt'),
            max_seq_length=MAX_SEQ_LENGTH,
            batch_size=BATCH_SIZE,
            label_ids=label_ids)

We need to create the classifier to sit on top of the pretrained model and define the loss function:

    .. code-block:: python

        hidden_size = bert_model.hidden_size
        ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                              num_classes=num_classes,
                                              dropout=CLASSIFICATION_DROPOUT)

        ner_loss = nemo_nlp.TokenClassificationLoss(d_model=hidden_size,
                                                num_classes=num_classes,
                                                dropout=CLASSIFICATION_DROPOUT)

Now, create the train and evaluation datasets:

    .. code-block:: python

        input_ids, input_type_ids, input_mask, loss_mask, _, labels = train_data_layer()

        hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)

        logits = ner_classifier(hidden_states=hidden_states)
        loss = ner_loss(logits=logits, labels=labels, loss_mask=loss_mask)


        eval_input_ids, eval_input_type_ids, eval_input_mask, _, eval_subtokens_mask, eval_labels \
        = eval_data_layer()

        hidden_states = bert_model(
            input_ids=eval_input_ids,
            token_type_ids=eval_input_type_ids,
            attention_mask=eval_input_mask)

        eval_logits = ner_classifier(hidden_states=hidden_states)

Now, we will set up our callbacks. We will use 3 callbacks:

* `SimpleLossLoggerCallback` to print loss values during training
* `EvaluatorCallback` to evaluate our F1 score on the dev dataset. In this example, `EvaluatorCallback` will also output predictions to `output.txt`, which can be helpful with debugging what our model gets wrong.
* `CheckpointCallback` to save and restore checkpoints.

    .. code-block:: python

        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss],
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())))

        train_data_size = len(train_data_layer)

        # If you're training on multiple GPUs, this should be
        # train_data_size / (batch_size * batches_per_step * num_gpus)
        steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_logits, eval_labels, eval_subtokens_mask],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, label_ids),
            eval_step=steps_per_epoch)

        # Callback to store checkpoints
        # Checkpoints will be stored in checkpoints folder inside WORK_DIR
        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=1)

Finally, we will define our learning rate policy and our optimizer, and start training.

    .. code-block:: python

        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 lr_policy=lr_policy,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

.. tip::
    
    Tensorboard_ is a great debugging tool. It's not a requirement for this tutorial, but if you'd like to use it, you should install tensorboardX_ and run the following command during fine-tuning:

    .. code-block:: bash
    
        tensorboard --logdir output_ner/tensorboard

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX

To train NER with BERT using the provided scripts
-------------------------------------------------

To run the provided training script:

.. code-block:: bash

    python token_classification.py --data_dir /data/ner/ --work_dir output_ner

To run inference:

.. code-block:: bash

    python token_classification_infer.py --labels_dict /data/ner/label_ids.csv
    --work_dir output_ner/checkpoints/

Note, label_ids.csv file will be generated during training and stored in the data_dir folder.

Using Other BERT Models
-----------------------

In addition to using pre-trained BERT models from Google and BERT models that you've trained yourself, in NeMo it's possible to use other third-party BERT models as well, as long as the weights were exported with PyTorch. For example, if you want to fine-tune an NER task with SciBERT_.

.. _SciBERT: https://github.com/allenai/scibert

.. code-block:: bash

    wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar
    tar -xf scibert_scivocab_cased.tar
    cd scibert_scivocab_cased
    tar -xzf weights.tar.gz
    mv bert_config.json config.json
    cd ..

And then, when you load your BERT model, you should specify the name of the directory for the model name.

.. code-block:: python

    tokenizer = NemoBertTokenizer(pretrained_model="scibert_scivocab_cased")
    bert_model = nemo_nlp.huggingface.BERT(
        pretrained_model_name="scibert_scivocab_cased")
