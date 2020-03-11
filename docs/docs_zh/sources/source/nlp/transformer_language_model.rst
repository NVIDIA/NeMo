Transformer语言模型
===================

在这个教程中，我们会用Transformer :cite:`nlp-lm-vaswani2017attention` 的结构构建和训练一个语言模型。
确保在开始这个教程之前你已经安装了 ``nemo`` 和 ``nemo_nlp`` ，详见 :ref:`installation` 。

简介
----

一个好的语言模型对于下游任务有很广泛的应用。用于下游任务的语言模型例子包括 GPT-2 :cite:`nlp-lm-radford2019language` 。


下载语料
--------

在这个实验中我们会使用非常小的WikiText-2数据集 :cite:`nlp-lm-merity2016pointer` 。

下载数据集，运行脚本 ``examples/nlp/language_modeling/get_wkt2.sh``. 下载和解压数据集后，文件夹会包括三个文件:

    .. code-block:: bash

        test.txt
        train.txt
        valid.txt

创建分词器模型
----------------
`LanguageModelDataDesc` 会把你的数据集转换到和 `LanguageModelingDataset` 兼容的格式。

    .. code-block:: python

        from nemo.collections.nlp.data.datasets.lm_transformer_dataset import LanguageModelDataDesc
        data_desc = LanguageModelDataDesc(
            args.dataset_name, args.data_dir, args.do_lower_case)

我们需要定义我们的分词器， 我们用定义在 ``nemo/collections/nlp/data/tokenizers/word_tokenizer.py`` 中的 `WordTokenizer`:

    .. code-block:: python

        import nemo.collections.nlp as nemo_nlp
        tokenizer = nemo_nlp.WordTokenizer(f"{args.data_dir}/{args.tokenizer_model}")
        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

    .. tip::
        让词嵌入的大小（或者其他张量的维度）能够整除 8 
        会帮助得到最好的 GPU 利用率，以及混精度训练的加速。

创建模型
----------------
首先我们需要用支持的后端来创建 ``neural factory`` 。你如何定义它取决于你想做多 GPU 训练或者是混合精度训练。
这个教程假设你不用混合精度，在一块 GPU 上做训练。如果你想做混合精度训练，设置 ``amp_opt_level`` 为 ``O1`` 或者 ``O2`` 。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

接着，我们定义对于我们模型的神经元模块

    * Transformer编码器 (注意，我们的语言模型不需要解码器)
    * `TokenClassifier` 把输出映射到词汇表上的概率分布.
    * 损失函数 (带标签平滑正则的交叉熵).

    .. code-block:: python

        from nemo.collections.nlp.nm.trainables.common import TokenClassifier
        from nemo.collections.nlp.nm.losses import SmoothedCrossEntropyLoss

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

        log_softmax = TokenClassifier(
            args.d_model, num_classes=vocab_size, num_layers=1, log_softmax=True
        )

        loss = SmoothedCrossEntropyLoss(pad_id=tokenizer.pad_id, label_smoothing=args.label_smoothing)

根据 `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`nlp-lm-press2016using`, 我们也会把词嵌入的参数和 softmax 层连起来:

    .. code-block:: python

        from nemo.core import WeightShareTransform
        log_softmax.tie_weights_with(
            encoder,
            weight_names=["mlp.layer0.weight"],
            name2name_and_transform={
                "mlp.layer0.weight": ("embedding_layer.token_embedding.weight", WeightShareTransform.SAME)
            },
        )

接着，我们创建从输入到输出的管道，用作训练和评估:

    .. code-block:: python

        from nemo.collections.nlp.nm.data_layers import LanguageModelingDataLayer

        def create_pipeline(
            dataset, max_seq_length=args.max_seq_length, batch_step=args.max_seq_length, batch_size=args.batch_size
        ):
            data_layer = LanguageModelingDataLayer(
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

接下来，我们定义一些必要的回调:

1. `SimpleLossLoggerCallback`: 记录训练中的 loss
2. `EvaluatorCallback`: 在用户设置的间隔中，追踪评估的度量指标
3. `CheckpointCallback`: 根据设定的时间点保存权重文件

    .. code-block:: python

        from nemo.collections.nlp.callbacks.lm_transformer_callback import eval_epochs_done_callback, eval_iter_callback
        train_callback = SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: logging.info(str(round(x[0].item(), 3))),
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

最后，定义优化器，开始训练吧！

    .. code-block:: python

        from nemo.utils.lr_policies import CosineAnnealing

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

参考
----

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-LM
    :keyprefix: nlp-lm-
