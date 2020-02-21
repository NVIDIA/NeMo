Transformer Language Model
==========================

In this tutorial, we will build and train a language model using the Transformer architecture :cite:`nlp-lm-vaswani2017attention`.
Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this tutorial. See the :ref:`installation` section for more details.

Introduction
------------

A good language model has a wide range of applications on downstream tasks. Examples of language models being used for downstream tasks include GPT-2 :cite:`nlp-lm-radford2019language`.


Download Corpus
---------------

For demonstration purposes, we will be using the very small WikiText-2 dataset :cite:`nlp-lm-merity2016pointer`.

To download the dataset, run the script ``examples/nlp/scripts/get_wkt2.sh``. After downloading and unzipping, the folder should include 3 files that look like this:

    .. code-block:: bash

        test.txt
        train.txt
        valid.txt

Create the tokenizer model
--------------------------
`LanguageModelDataDesc` converts your dataset into the format compatible with `LanguageModelingDataset`.

    .. code-block:: python

        data_desc = LanguageModelDataDesc(
            args.dataset_name, args.data_dir, args.do_lower_case)

We need to define our tokenizer. We use `WordTokenizer` defined in ``nemo/collections/nlp/data/tokenizers/word_tokenizer.py``:

    .. code-block:: python

        tokenizer = nemo_nlp.WordTokenizer(f"{args.data_dir}/{args.tokenizer_model}")
        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

    .. tip::
        Making embedding size (as well as all other tensor dimensions) divisible
        by 8 will help to get the best GPU utilization and speed-up with mixed precision training.

Create the model
----------------
First, we need to create our neural factory with the supported backend. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training.
This tutorial assumes that you're training on one GPU, without mixed precision. If you want to use mixed precision, set ``amp_opt_level`` to ``O1`` or ``O2``.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

Next, we define all Neural Modules necessary for our model 

    * Transformer Encoder (note that we don't need a decoder for language modeling)
    * `TokenClassifier` for mapping output of the decoder into probability distribution over vocabulary.
    * Loss function (cross entropy with label smoothing regularization).

    .. code-block:: python

        encoder = nemo_nlp.nm.trainables.TransformerEncoderNM(
            d_model=args.d_model,
            d_inner=args.d_inner,
            num_layers=args.num_layers,
            embedding_dropout=args.embedding_dropout,
            num_attn_heads=args.num_attn_heads,
            ffn_dropout=args.ffn_dropout,
            vocab_size=vocab_size,
            mask_future=True,
            attn_score_dropout=args.attn_score_dropout,
            attn_layer_dropout=args.attn_layer_dropout,
            max_seq_length=args.max_seq_length,
        )

        log_softmax = nemo.collections.nlp.nm.trainables.TokenClassifier(
            args.d_model, num_classes=vocab_size, num_layers=1, log_softmax=True
        )

        loss = nemo_nlp.nm.losses.SmoothedCrossEntropyLoss(pad_id=tokenizer.pad_id, label_smoothing=args.label_smoothing)

Following `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`nlp-lm-press2016using`, we also tie the parameters of embedding and softmax layers:

    .. code-block:: python

        log_softmax.tie_weights_with(
            encoder,
            weight_names=["mlp.layer0.weight"],
            name2name_and_transform={
                "mlp.layer0.weight": ("embedding_layer.token_embedding.weight", WeightShareTransform.SAME)
            },
        )

Then, we create the pipeline from input to output that can be used for both training and evaluation:

    .. code-block:: python

        def create_pipeline(
            dataset, max_seq_length=args.max_seq_length, batch_step=args.max_seq_length, batch_size=args.batch_size
        ):
            data_layer = nemo.collections.nlp.nm.data_layers.LanguageModelingDataLayer(
                dataset, tokenizer, max_seq_length, batch_size, batch_step
            )
            src, src_mask, labels = data_layer()
            src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
            logits = log_softmax(hidden_states=src_hiddens)
            return loss(logits=logits, labels=labels)


        train_loss = create_pipeline(
            f"{args.data_dir}/{args.train_dataset}",
            args.max_seq_length,
            batch_step=args.max_seq_length,
            batch_size=args.batch_size,
        )
        eval_loss = create_pipeline(
            f"{args.data_dir}/{args.eval_dataset}",
            args.max_seq_length,
            batch_step=args.predict_last_k,
            batch_size=args.eval_batch_size,
        )

Next, we define necessary callbacks:

1. `SimpleLossLoggerCallback`: tracking loss during training
2. `EvaluatorCallback`: tracking metrics during evaluation at set intervals
3. `CheckpointCallback`: saving model checkpoints at set intervals

    .. code-block:: python

        train_callback = SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: str(np.round(x[0].item(), 3)),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=steps_per_epoch,
        )

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y, data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch,
        )

        # Create callback to save checkpoints
        ckpt_callback = CheckpointCallback(
            folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
        )

Finally, you should define your optimizer, and start training!

    .. code-block:: python

        lr_policy_fn = CosineAnnealing(args.max_steps, warmup_steps=args.warmup_steps)
        max_num_epochs = 0 if args.interactive else args.num_epochs

        callbacks = [callback_ckpt]
        if not args.interactive:
            callbacks.extend([train_callback, eval_callback])

        nf.train(
            tensors_to_optimize=[train_loss],
            callbacks=callbacks,
            lr_policy=lr_policy_fn,
            batches_per_step=args.iter_per_step,
            optimizer=args.optimizer_kind,
            optimization_params={
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": (args.beta1, args.beta2),
            },
        )

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-LM
    :keyprefix: nlp-lm-
