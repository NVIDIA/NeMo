Tutorial
========

Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this
tutorial. See the :ref:`installation` section for more details.

Introduction
------------

This tutorial explains how to implement named entity recognition (NER) in NeMo. We'll show how to do this with a pre-trained BERT model, or with one that you trained yourself! For more details, check out our BERT pretraining tutorial.

Download Dataset
----------------

`CoNLL-2003`_ is a standard evaluation dataset for NER, but any NER dataset will work. The only requirement is that the files are formatted like this:

.. _CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/

.. code-block::

    Jennifer	B-PER
    is		O
    from	O
    New		B-LOC
    York	I-LOC
    City	I-LOC
    .		O

    She		O
    likes	O
    ...

Here, the words and labels are separated with spaces, but in your dataset they should be separated with tabs. Each line should follow the format: [WORD] [TAB] [LABEL] (without spaces in between). There can be columns in between for part-of-speech tags, as shown on the `CoNLL-2003 website`_. There should also be empty lines separating each sequence, as shown above.

.. _CoNLL-2003 website: https://www.clips.uantwerpen.be/conll2003/ner/

.. _Preprocessed data: https://github.com/kyzhouhzau/BERT-NER/tree/master/data

Training
--------

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!

First, we need to create our neural factory with the supported backend. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision. If you want to use mixed precision, set ``amp_opt_level`` to ``O1`` or ``O2``.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

Next, we'll need to define our tokenizer and our BERT model. There are a couple of different ways you can do this. Keep in mind that NER benefits from casing ("New York City" is easier to identify than "new york city"), so we recommend you use cased models.

If you're using a standard BERT model, you should do it as follows. To see the full list of BERT model names, check out ``nemo_nlp.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
        pretrained_bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name=args.pretrained_bert_model)

If you're using a BERT model that you pre-trained yourself, you should do it like this. You should replace ``args.bert_checkpoint`` with the path to your checkpoint file.

    .. code-block:: python

        tokenizer = SentencePieceTokenizer(model_path=tokenizer_model)
        tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

        bert_model = nemo_nlp.huggingface.BERT(
                config_filename=args.bert_config)
        pretrained_bert_model.restore_from(args.bert_checkpoint)

Now, create the train and evaluation datasets:

    .. code-block:: python

    train_data_layer = nemo_nlp.data.BertTokenClassificationDataLayer(
        dataset_type="BertCornellNERDataset",
        tokenizer=tokenizer,
        input_file=os.path.join(DATA_DIR, "train.txt"),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE)

    eval_data_layer = nemo_nlp.data.BertTokenClassificationDataLayer(
        dataset_type="BertCornellNERDataset",
        tokenizer=tokenizer,
        input_file=os.path.join(DATA_DIR, "dev.txt"),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE)

We need to create the classifier to sit on top of the pretrained model and define the loss function:

    .. code-block:: python

        hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
        tag_ids = train_dataset.tag_ids
        ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                                  num_classes=len(tag_ids),
                                                  dropout=args.fc_dropout)
        ner_loss = nemo_nlp.TokenClassificationLoss(num_classes=len(tag_ids))

And create the pipeline that can be used for both training and evaluation.

    .. code-block:: python

        def create_pipeline(data_layer, batch_size=args.batch_size,
                            local_rank=args.local_rank, num_gpus=args.num_gpus):
            input_ids, input_type_ids, input_mask, labels, seq_ids = data_layer()
            hidden_states = pretrained_bert_model(input_ids=input_ids,
                                                  token_type_ids=input_type_ids,
                                                  attention_mask=input_mask)
            logits = ner_classifier(hidden_states=hidden_states)
            loss = ner_loss(logits=logits, labels=labels, input_mask=input_mask)
            steps_per_epoch = len(data_layer) // (batch_size * num_gpus)
            return loss, steps_per_epoch, data_layer, [logits, seq_ids]

        train_loss, steps_per_epoch, _, _ = create_pipeline(train_data_layer)
        _, _, data_layer, eval_tensors = create_pipeline(eval_data_layer)

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
            tensors=[train_loss],
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            tb_writer=nf.tb_writer)

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(
                x, y, data_layer, tag_ids),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x, tag_ids, output_file),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=args.save_epoch_freq,
            step_freq=args.save_step_freq)

Finally, we will define our learning rate policy and our optimizer, and start training.

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)


        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 lr_policy=lr_policy_fn,
                 optimizer=args.optimizer_kind,
                 optimization_params={"num_epochs": args.num_epochs,
                                      "lr": args.lr})

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
