Transformer语言模型
===================

在这个教程中，我们会用Transformer :cite:`nlp-lm-vaswani2017attention` 的结构构建和训练一个语言模型。确保在开始这个教程之前你已经安装了 ``nemo`` 和 ``nemo_nlp`` ，详见 :ref:`installation` 。

简介
------------

一个好的语言模型对于下游任务有很广泛的应用。用于下游任务的语言模型例子包括 GPT-2 :cite:`nlp-lm-radford2019language` 。

下载语料
---------------

在这个实验中我们会使用非常小的WikiText-2数据集 :cite:`nlp-lm-merity2016pointer` 。

下载数据集，运行脚本 ``examples/nlp/scripts/get_wt2.sh``. 下载和解压数据集后，文件夹会包括三个文件:

    .. code-block:: bash

        test.txt
        train.txt
        valid.txt

创建分词器模型
----------------
`LanguageModelDataDesc` 会把你的数据集转换到和 `LanguageModelingDataset` 兼容的格式。

    .. code-block:: python

        data_desc = LanguageModelDataDesc(
            args.dataset_name, args.data_dir, args.do_lower_case)

我们需要定义我们的分词器， 我们用定义在 ``nemo_nlp/data/tokenizers/word_tokenizer.py`` 中的 `WordTokenizer`:

    .. code-block:: python

        tokenizer = nemo_nlp.WordTokenizer(f"{args.data_dir}/{args.tokenizer_model}")
        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

    .. tip::
        让词嵌入的大小（或者其他张量的维度）能够整除8会帮助得到最好的GPU利用率，以及混精度训练的加速。

创建模型
----------------
首先我们需要用支持的后端来创建 ``neural factory`` 。你如何定义它取决于你想做多GPU训练或者是混精度训练。这个教程假设你不用混精度，在一块GPU上做训练。
如果你想做混精度训练，设置 ``amp_opt_level`` 为 ``O1`` 或者 ``O2`` 。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

接着，我们定义对于我们模型的神经元模块

    * Transformer编码器 (注意，我们的语言模型不需要解码器)
    * `TokenClassifier`  把输出映射到词汇表上的概率分布.
    * 损失函数 (带标签平滑正则的交叉熵).

    .. code-block:: python

        encoder = nemo_nlp.TransformerEncoderNM(**params)
        log_softmax = nemo_nlp.TokenClassifier(**params)
        loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(**params)


根据 `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`nlp-lm-press2016using`, 我们也会把词嵌入的参数和softmax层连起来:

    .. code-block:: python

        log_softmax.mlp.layers[-1].weight = encoder.embedding_layer.token_embedding.weight


接着，我们为训练和评估创建数据集:

    .. code-block:: python

        train_dataset = nemo_nlp.LanguageModelingDataset(
            tokenizer,
            dataset=f"{args.data_dir}/{args.train_dataset}",
            max_sequence_length=args.max_sequence_length,
            batch_step=args.max_sequence_length)

        eval_dataset = nemo_nlp.LanguageModelingDataset(
            tokenizer,
            dataset=f"{args.data_dir}/{args.eval_datasets[0]}",
            max_sequence_length=args.max_sequence_length,
            batch_step=args.predict_last_k)


然后,我们创建用于训练和评估的从输入到输出的管道:

    .. code-block:: python

        def create_pipeline(dataset, batch_size):
            data_layer = nemo_nlp.LanguageModelingDataLayer(dataset,
                                                            batch_size=batch_size)
            src, src_mask, labels = data_layer()
            src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
            logits = log_softmax(hidden_states=src_hiddens)
            return loss(logits=logits, target_ids=labels)


        train_loss = create_pipeline(train_dataset, args.batch_size)
        eval_loss = create_pipeline(eval_dataset, args.batch_size)
    

接下来，我们定义一些必要的回调:

1. `SimpleLossLoggerCallback`: 追踪训练中的 loss
2. `EvaluatorCallback`: 在用户设置的间隔中，追踪评估的度量指标
3. `CheckpointCallback`: 在设置的间各种保存 checkpoints

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(...)
        eval_callback = nemo.core.EvaluatorCallback(...)
        ckpt_callback = nemo.core.CheckpointCallback(...)


最后，定义优化器，开始训练吧！

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=callbacks,
                 lr_policy=lr_policy_fn,
                 batches_per_step=args.iter_per_step,
                 optimizer=args.optimizer_kind,
                 optimization_params={"num_epochs": args.num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay,
                                      "betas": (args.beta1, args.beta2)})

参考
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-LM
    :keyprefix: nlp-lm-
