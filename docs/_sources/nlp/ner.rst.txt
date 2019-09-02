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

Training
--------

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!

Here, we'll fine-tune a BERT model on our downstream NER task. We'll start off with our imports and constants.

.. code-block:: python

    import math
    import os

    import nemo
    from nemo.utils.lr_policies import WarmupAnnealing

    import nemo_nlp
    from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
    from nemo_nlp.callbacks.ner import \
        eval_iter_callback, eval_epochs_done_callback

    BATCHES_PER_STEP = 1
    BATCH_SIZE = 32
    CLASSIFICATION_DROPOUT = 0.1
    DATA_DIR = "conll2003"
    MAX_SEQ_LENGTH = 128
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.00005
    LR_WARMUP_PROPORTION = 0.1
    OPTIMIZER = "adam"

Next, we need to create our neural factory. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision.

.. code-block:: python

    # Instantiate neural factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,

        # If you're training with multiple GPUs, you should handle this value with
        # something like argparse. See examples/nlp/ner.py for an example.
        local_rank=None,

        # If you're training with mixed precision, this should be set to mxprO1 or mxprO2.
        # See https://nvidia.github.io/apex/amp.html#opt-levels for more details.
        optimization_level=nemo.core.Optimization.mxprO0,

        # If you're training with multiple GPUs, this should be set to
        # nemo.core.DeviceType.AllGpu
        placement=nemo.core.DeviceType.GPU)

Next, we'll need to define our tokenizer and our BERT model. There are a couple of different ways you can do this. Keep in mind that NER benefits from casing ("New York City" is easier to identify than "new york city"), so we recommend you use cased models.

.. code-block:: python

    # If you're using a standard BERT model, you should do it like this. To see the full
    # list of BERT model names, check out nemo_nlp.huggingface.BERT.list_pretrained_models()
    tokenizer = NemoBertTokenizer(pretrained_model="bert-base-cased")
    bert_model = nemo_nlp.huggingface.BERT(
        pretrained_model_name="bert-base-cased",
        factory=neural_factory)

    # If you're using a BERT model that you pre-trained yourself, you should do it like this.
    # You should replace BERT-STEP-150000.pt with the path to your checkpoint file.
    tokenizer = SentencePieceTokenizer(model_path="tokenizer.model")
    tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

    bert_model = nemo_nlp.huggingface.BERT(
        config_filename=os.path.join("bert_pretraining_checkpoints", "config.json"),
        factory=neural_factory)
    bert_model.restore_from(
        os.path.join("bert_pretraining_checkpoints", "BERT-STEP-150000.pt"))

Now, we will define the training pipeline:

.. code-block:: python

    train_data_layer = nemo_nlp.BertNERDataLayer(
        tokenizer=tokenizer,
        path_to_data=os.path.join(DATA_DIR, "train.txt"),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        factory=neural_factory)

    tag_ids = train_data_layer.dataset.tag_ids

    ner_loss = nemo_nlp.TokenClassificationLoss(
        d_model=bert_model.bert.config.hidden_size,
        num_labels=len(tag_ids),
        dropout=CLASSIFICATION_DROPOUT,
        factory=neural_factory)

    input_ids, input_type_ids, input_mask, labels, _ = train_data_layer()

    hidden_states = bert_model(
        input_ids=input_ids,
        token_type_ids=input_type_ids,
        attention_mask=input_mask)

    train_loss, train_logits = ner_loss(
        hidden_states=hidden_states,
        labels=labels,
        input_mask=input_mask)

And now, our evaluation pipeline:

.. code-block:: python

    eval_data_layer = nemo_nlp.BertNERDataLayer(
        tokenizer=tokenizer,
        path_to_data=os.path.join(DATA_DIR, "dev.txt"),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        factory=neural_factory)

    input_ids, input_type_ids, eval_input_mask, \
        eval_labels, eval_seq_ids = eval_data_layer()

    hidden_states = bert_model(
        input_ids=input_ids,
        token_type_ids=input_type_ids,
        attention_mask=eval_input_mask)

    eval_loss, eval_logits = ner_loss(
        hidden_states=hidden_states,
        labels=eval_labels,
        input_mask=eval_input_mask)

Now, we will set up our callbacks. Here, we will use `SimpleLossLoggerCallback` to print loss values during training, and `EvaluatorCallback` to evaluate our F1 score on the dev dataset. In this example, `EvaluatorCallback` will also output predictions to `output.txt`, which can be helpful with debugging what our model gets wrong.

.. tip::
    
    Tensorboard_ is a great debugging tool. It's not a requirement for this tutorial, but if you'd like to use it, you should install tensorboardX_ and run the following command during fine-tuning:

    .. code-block:: bash
    
        tensorboard --logdir bert_ner_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX

.. code-block:: python

    try:
        import tensorboardX
        tb_writer = tensorboardX.SummaryWriter("bert_ner_tb")
    except ModuleNotFoundError:
        tb_writer = None
        print("Tensorboard is not available")

    callback_train = nemo.core.SimpleLossLoggerCallback(
        tensors=[train_loss],
        print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
        get_tb_values=lambda x: [["loss", x[0]]],
        tb_writer=tb_writer)

    train_data_size = len(train_data_layer)

    # If you're training on multiple GPUs, this should be
    # train_data_size / (batch_size * batches_per_step * num_gpus)
    steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

    callback_eval = nemo.core.EvaluatorCallback(
        eval_tensors=[eval_logits, eval_seq_ids],
        user_iter_callback=lambda x, y: eval_iter_callback(
            x, y, eval_data_layer, tag_ids),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(
            x, tag_ids, "output.txt"),
        tb_writer=tb_writer,
        eval_step=steps_per_epoch)

Finally, we will define our learning rate policy and our optimizer, and start training.

.. code-block:: python

    lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                                warmup_ratio=LR_WARMUP_PROPORTION)
    optimizer = neural_factory.get_trainer()
    optimizer.train(
        tensors_to_optimize=[train_loss],
        callbacks=[callback_train, callback_eval],
        lr_policy=lr_policy,
        batches_per_step=BATCHES_PER_STEP,
        optimizer=OPTIMIZER,
        optimization_params={
            "num_epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE
        })

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
