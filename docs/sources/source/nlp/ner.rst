Tutorial
========

Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this
tutorial. See the :ref:`installation` section for more details.

Introduction
------------

This tutorial explains how to implement named entity recognition (NER) in NeMo. We'll show how to do this with a pre-trained BERT model, or with one that you trained yourself! For more details, check out our BERT pretraining tutorial.

Download Dataset
----------------

`CoNLL-2003`_ is a standard evaluation dataset for NER, but any NER dataset will work. CoNLL-2003 dataset could also be found `here`_. The only requirement is that the data is splitted into 2 files: text.txt and labels.txt. The text.txt files should be formatted like this:

.. _CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/
.. _here: https://github.com/kyzhouhzau/BERT-NER/tree/master/data

.. code-block::

    Jennifer is from New York City.
    She likes ...
    ...

The labels.txt files should be formatted like this:

.. code-block::

    B-PER O O B-LOC I-LOC I-LOC O
    O O ...
    ...

Each line of the text.txt file contains text sequences, where words are separated with spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces. Each line of the files should follow the format: [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). There can be columns in between for part-of-speech tags, as shown on the `CoNLL-2003 website`_.

.. _CoNLL-2003 website: https://www.clips.uantwerpen.be/conll2003/ner/



Training
--------

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!
    See examples/nlp/NERWithBERT.ipynb

First, we need to create our neural factory with the supported backend. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision. If you want to use mixed precision, set ``amp_opt_level`` to ``O1`` or ``O2``.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=None,
                                           optimization_level=nemo.core.Optimization.mxprO0,
                                           create_tb_writer=True)

Next, we'll need to define our tokenizer and our BERT model. There are a couple of different ways you can do this. Keep in mind that NER benefits from casing ("New York City" is easier to identify than "new york city"), so we recommend you use cased models.

If you're using a standard BERT model, you should do it as follows. To see the full list of BERT model names, check out ``nemo_nlp.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = NemoBertTokenizer(pretrained_model="bert-base-cased")
        bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-cased",
            factory=neural_factory)

If you're using a BERT model that you pre-trained yourself, you should do it like this. You should replace ``args.bert_checkpoint`` with the path to your checkpoint file.

    .. code-block:: python

        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model)
        tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

        bert_model = nemo_nlp.huggingface.BERT(
                config_filename=args.bert_config)
        pretrained_bert_model.restore_from(args.bert_checkpoint)

We need to create the classifier to sit on top of the pretrained model and define the loss function:

    .. code-block:: python

        hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
        
        ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                                  num_classes=NUM_CLASSES,
                                                  dropout=CLASSIFICATION_DROPOUT)
        ner_loss = nemo_nlp.TokenClassificationLoss(num_classes=NUM_CLASSES)

And create the pipeline that can be used for both training and evaluation.

    .. code-block:: python

        def create_pipeline(max_seq_length=MAX_SEQ_LENGTH,
                            batch_size=BATCH_SIZE,
                            mode='train'):
        
        text_file = f'{DATA_DIR}/text_{mode}.txt'
        label_file = f'{DATA_DIR}/labels_{mode}.txt'
        
        data_layer = nemo_nlp.BertTokenClassificationDataLayer(
            tokenizer=tokenizer,
            text_file=text_file,
            label_file=label_file,
            max_seq_length=max_seq_length,
            batch_size=batch_size)

        label_ids = data_layer.dataset.label_ids
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = data_layer()
        hidden_states = bert_model(input_ids=input_ids,
                                   token_type_ids=input_type_ids,
                                   attention_mask=input_mask)

        logits = classifier(hidden_states=hidden_states)
        loss = punct_loss(logits=logits, labels=labels, loss_mask=loss_mask)
        steps_per_epoch = len(data_layer) // (batch_size * num_gpus)

        if mode == 'train':
             tensors_to_evaluate = [loss, logits]
        else:
             tensors_to_evaluate = [logits, labels, subtokens_mask]
        return tensors_to_evaluate, loss, steps_per_epoch, label_ids, data_layer

Now, create the train and evaluation datasets:

.. code-block:: python
    train_tensors, train_loss, steps_per_epoch, label_ids, _ = create_pipeline()
    eval_tensors, _, _, _, data_layer = create_pipeline(mode='dev')

Now, we will set up our callbacks. We will use 3 callbacks:

* `SimpleLossLoggerCallback` to print loss values during training
* `EvaluatorCallback` to evaluate our F1 score on the dev dataset. In this example, `EvaluatorCallback` will also output predictions to `output.txt`, which can be helpful with debugging what our model gets wrong.
* `CheckpointCallback` to save and restore checkpoints.

.. tip::
    
    Tensorboard_ is a great debugging tool. It's not a requirement for this tutorial, but if you'd like to use it, you should install tensorboardX_ and run the following command during fine-tuning:

    .. code-block:: bash
    
        tensorboard --logdir bert_ner_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            tb_writer=nf.tb_writer)

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x:
                eval_epochs_done_callback(x, label_ids),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

Finally, we will define our learning rate policy and our optimizer, and start training.

    .. code-block:: python

        
        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=[train_callback, eval_callback],
                 lr_policy=lr_policy,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

To train NEW with BERT using the provided scripts
-----------------------

To run the provided training script:

.. code-block:: bash

    python token_classification.py --num_classes 9 --data_dir /data/ner/ --work_dir output_ner

To run inference:

.. code-block:: bash

    python token_classification_infer.py --num_classes 9 --labels_dict /data/ner/label_ids.csv
    --work_dir output_ner/checkpoints/

Note, label_ids.csv file will be generated during training and stored in the data_dir folder.

Using Other BERT Models
-----------------------

In addition to using pre-trained BERT models from Google and BERT models that you've trained yourself, in NeMo it's possible to use other third-party BERT models as well, as long as the weights were exported with PyTorch. For example, if you want to fine-tune an NER task with SciBERT_...

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
        pretrained_model_name="scibert_scivocab_cased",
        factory=neural_factory)

If you want to use a TensorFlow-based model, such as BioBERT, you should be able to use it in NeMo by first using this `model conversion script`_ provided by Hugging Face.

.. _model conversion script: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_tf_checkpoint_to_pytorch.py
