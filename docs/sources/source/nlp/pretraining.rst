Pretraining BERT
================

Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this
tutorial. See the :ref:`installation` section for more details.

Introduction
------------

This tutorial is focused on pretraining BERT from scratch. Creating domain-specific BERT models can be advantageous for a wide range of applications. Most notably, in a biomedical setting, similar to BioBERT :cite:`lee2019biobert` and SciBERT :cite:`beltagy2019scibert`.

Download Corpus
---------------

For demonstration purposes, we will be using the very small WikiText-2 corpus. This script will download and unzip the corpus for you:

.. code-block:: bash

   ./tests/data/get_wt2.sh

After the download has completed, there should be a `wikitext-2` folder in your current directory, which should include `train.txt`, `valid.txt`, and `test.txt`.

Build Vocabulary
----------------

.. note::
    This step is optional! If you don't want to use a custom vocabulary, using the `vocab.txt` file from any `pretrained BERT model`_ will do. Also, depending on the size of your corpus, this may take awhile.

.. _pretrained BERT model: https://github.com/google-research/bert#pre-trained-models

Another script can be used to generate your vocabulary file. In this example with WikiText-2, you can build it like this:

.. code-block:: bash

    # In this example, our dataset consists of one file, so we can run it like this:
    python tests/data/create_vocab.py --train_path wikitext-2/train.txt

    # If your corpus consists of many different files, you should run it like this instead:
    python tests/data/create_vocab.py --dataset_dir path_to_dataset/

The script will output two important files: `tokenizer.vocab` and `tokenizer.model`. We'll explain how to use both in the next section.

Training
--------

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!

Here, we will pre-train a BERT model from scratch on the WikiText-2 corpus. We'll start off with our imports and constants.

.. code-block:: python

    import math
    import os

    import nemo
    from nemo.utils.lr_policies import CosineAnnealing

    import nemo_nlp
    from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
    from nemo_nlp.callbacks.bert_pretraining import eval_iter_callback, \
        eval_epochs_done_callback

    BATCHES_PER_STEP = 1
    BATCH_SIZE = 64
    BATCH_SIZE_EVAL = 16
    CHECKPOINT_DIR = "bert_pretraining_checkpoints"
    D_MODEL = 768
    D_INNER = 3072
    HIDDEN_ACT = "gelu"
    LEARNING_RATE = 1e-2
    LR_WARMUP_PROPORTION = 0.05
    MASK_PROBABILITY = 0.15
    MAX_SEQ_LENGTH = 128
    NUM_EPOCHS = 10
    NUM_HEADS = 12
    NUM_LAYERS = 12
    OPTIMIZER = "novograd"
    WEIGHT_DECAY = 0

Next, we need to create our neural factory. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision.

.. code-block:: python

    # Instantiate neural factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,

        # If you're training with multiple GPUs, you should handle this value with
        # something like argparse. See examples/nlp/bert_pretraining.py for an example.
        local_rank=None,

        # If you're training with mixed precision, this should be set to mxprO1 or mxprO2.
        # See https://nvidia.github.io/apex/amp.html#opt-levels for more details.
        optimization_level=nemo.core.Optimization.mxprO0,

        # If you're training with multiple GPUs, this should be set to
        # nemo.core.DeviceType.AllGpu
        placement=nemo.core.DeviceType.GPU)

Now, we need to define our tokenizer. If you'd like to use a custom vocabulary file, we strongly recommend you use our `SentencePieceTokenizer`. Otherwise, if you'll be using a vocabulary file from another pre-trained BERT model, you should use `NemoBertTokenizer`.

.. code-block:: python

    # If you're using a custom vocabulary, create your tokenizer like this
    tokenizer = SentencePieceTokenizer(model_path="tokenizer.model")
    tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

    # Otherwise, create your tokenizer like this
    tokenizer = NemoBertTokenizer(vocab_file="vocab.txt")

We also need to define the BERT model that we will be pre-training. Here, you can configure your model size as needed.

.. code-block:: python

    bert_model = nemo_nlp.huggingface.BERT(
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=D_MODEL,
        num_attention_heads=NUM_HEADS,
        intermediate_size=D_INNER,
        max_position_embeddings=MAX_SEQ_LENGTH,
        hidden_act=HIDDEN_ACT,
        factory=neural_factory)

    # If you want to start pre-training from existing BERT checkpoints, you should create
    # the model like this instead. For the full list of BERT model names, check out
    # nemo_nlp.huggingface.BERT.list_pretrained_models()
    bert_model = nemo_nlp.huggingface.BERT(
        pretrained_model_name="bert-base-cased",
        factory=neural_factory)

Next, we will define our loss functions. We will demonstrate how to pre-train with both MLM and NSP losses, but you may observe higher downstream accuracy by only pre-training with MLM loss.

.. code-block:: python

    mlm_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        factory=neural_factory)
    mlm_loss = nemo_nlp.MaskedLanguageModelingLossNM(factory=neural_factory)

    mlm_log_softmax.log_softmax.dense.weight = \
        bert_model.bert.embeddings.word_embeddings.weight

    nsp_log_softmax = nemo_nlp.SentenceClassificationLogSoftmaxNM(
        d_model=D_MODEL,
        num_classes=2,
        factory=neural_factory)
    nsp_loss = nemo_nlp.NextSentencePredictionLossNM(factory=neural_factory)

    bert_loss = nemo_nlp.LossAggregatorNM(
        num_inputs=2,
        factory=neural_factory)

Another crucial pre-training component is our data layer. If you're training on larger corpora, you can pass a directory name into the `dataset` argument, but we can do our example like this:

.. code-block:: python

    train_data_layer = nemo_nlp.BertPretrainingDataLayer(
        tokenizer=tokenizer,
        dataset=os.path.join("wikitext-2", "train.txt"),
        name="train",
        max_seq_length=MAX_SEQ_LENGTH,
        mask_probability=MASK_PROBABILITY,
        batch_size=BATCH_SIZE,
        factory=neural_factory)

    test_data_layer = nemo_nlp.BertPretrainingDataLayer(
        tokenizer=tokenizer,
        dataset=os.path.join("wikitext-2", "test.txt"),
        name="test",
        max_seq_length=MAX_SEQ_LENGTH,
        mask_probability=MASK_PROBABILITY,
        batch_size=BATCH_SIZE_EVAL,
        factory=neural_factory)

Next, we will define our training pipeline.

.. code-block:: python

    input_ids, input_type_ids, input_mask, \
        output_ids, output_mask, nsp_labels = train_data_layer()

    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)

    train_mlm_log_probs = mlm_log_softmax(hidden_states=hidden_states)
    train_mlm_loss = mlm_loss(log_probs=train_mlm_log_probs,
                              output_ids=output_ids,
                              output_mask=output_mask)

    train_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states)
    train_nsp_loss = nsp_loss(log_probs=train_nsp_log_probs, labels=nsp_labels)
    train_loss = bert_loss(loss_1=train_mlm_loss, loss_2=train_nsp_loss)

And testing pipeline.

.. code-block:: python

    input_ids_, input_type_ids_, input_mask_, \
        output_ids_, output_mask_, nsp_labels_ = test_data_layer()

    hidden_states_ = bert_model(input_ids=input_ids_,
                                token_type_ids=input_type_ids_,
                                attention_mask=input_mask_)

    test_mlm_log_probs = mlm_log_softmax(hidden_states=hidden_states_)
    test_mlm_loss = mlm_loss(log_probs=test_mlm_log_probs,
                             output_ids=output_ids_,
                             output_mask=output_mask_)

    test_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states_)
    test_nsp_loss = nsp_loss(log_probs=test_nsp_log_probs, labels=nsp_labels_)

Now, we will define our callbacks. NeMo provides a variety of callbacks for you to use; in this tutorial, we will make use of `SimpleLossLoggerCallback`, which prints loss values during training, `CheckpointCallback`, which saves model checkpoints at set intervals, and `EvaluatorCallback`, which evaluates test loss at set intervals.

.. tip::

    Tensorboard_ is a great debugging tool. It's not a requirement for this tutorial, but if you'd like to use it, you should install tensorboardX_ and run the following command during pre-training:

    .. code-block:: bash

        tensorboard --logdir bert_pretraining_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX

.. code-block:: python

    try:
        import tensorboardX
        tb_writer = tensorboardX.SummaryWriter("bert_pretraining_tb")
    except ModuleNotFoundError:
        tb_writer = None
        print("Tensorboard is not available")

    callback_loss = nemo.core.SimpleLossLoggerCallback(
        tensors=[train_loss],
        print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
        get_tb_values=lambda x: [["loss", x[0]]],
        tb_writer=tb_writer)

    callback_ckpt = nemo.core.CheckpointCallback(
        folder=CHECKPOINT_DIR,
        step_freq=25000)

    train_data_size = len(train_data_layer)

    # If you're training on multiple GPUs, this should be
    # train_data_size / (batch_size * batches_per_step * num_gpus)
    steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

    callback_test = nemo.core.EvaluatorCallback(
        eval_tensors=[test_mlm_loss, test_nsp_loss],
        user_iter_callback=eval_iter_callback,
        user_epochs_done_callback=eval_epochs_done_callback,
        eval_step=steps_per_epoch,
        tb_writer=tb_writer)

We also recommend you export your model's parameters to a config file. This makes it easier to load your BERT model into NeMo later, as explained in our NER tutorial.

.. code-block:: python

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(config_path):
        bert_model.config.to_json_file(config_path)

Finally, you should define your optimizer, and start training!

.. code-block:: python

    lr_policy = CosineAnnealing(NUM_EPOCHS * steps_per_epoch,
                                warmup_ratio=LR_WARMUP_PROPORTION)
    neural_factory.train(tensors_to_optimize=[train_loss],
                    lr_policy=lr_policy,
                    callbacks=[callback_loss, callback_ckpt, callback_test],
                    batches_per_step=BATCHES_PER_STEP,
                    optimizer=OPTIMIZER,
                    optimization_params={
                        "batch_size": BATCH_SIZE,
                        "num_epochs": NUM_EPOCHS,
                        "lr": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "betas": (0.95, 0.98),
                        "grad_norm_clip": None
                    })

References
----------

.. bibliography:: Bertbib.bib
    :style: plain
