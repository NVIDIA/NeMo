教程
====

在教程前，请确认你已经安装了 ``nemo`` 和 ``nemo_nlp`` 。你可以通过这个部分获得更多的信息 :ref:`installation`


简介
----

这个教程将介绍如何在NeMo中，实现命名实体识别(Named Entity Recognition, NER)。我们将通过一个预训练好的 BERT 模型来进行展示，或者你也可以使用一个训练好的模型！你可以通过 BERT 预训练教程获得更多的信息。

下载数据集
----------

`CoNLL-2003`_ 是一个NER上标准的验证集合，当然任何一个NER数据集合都可以。数据集合需要满足的条件是符合以下的格式:

.. _CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/

.. code-block:: python

    Jennifer    B-PER
    is      O
    from    O
    New     B-LOC
    York    I-LOC
    City    I-LOC
    .       O

    She     O
    likes   O
    ...

这里，词和标注使用空格进行分隔，但是在你的数据集里，它们需要用用tab分隔。每一行需要符合这样的格式: [WORD] [TAB] [LABEL] （中间没有空格)。一如 `CoNLL-2003 website`_ 网页上，词性标签之间可能会有列。每个句子中间需要用空行进行分隔。

.. _CoNLL-2003 website: https://www.clips.uantwerpen.be/conll2003/ner/

.. _Preprocessed data: https://github.com/kyzhouhzau/BERT-NER/tree/master/data

训练
----

.. tip::

    我们建议试试使用Jupyter来运行这部分代码，这会使得调试更加容易!

首先，我们需要使用所支持的后端，来创建我们的 `Neural Factory` 。你需要确认使用多GPU或者混合精度训练。这个教程中我们使用单GPU训练，不使用混合精度。如果你想使用混合精度训练，需要设置 ``amp_opt_level`` 这个参数为 ``O1`` 或者 ``O2`` 。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

接着，我们需要定义我们的分词器 (tokenizer) 和 BERT 模型。你可以有多种方式来实现。注意，NER是大小写敏感的（"New York City"比"new york city"更容易被识别出来），所以我们建议使用区分大小写的模型。

如果你正在使用一个标准的 BERT 模型，我们建议你使用下面这条命令。想获取完整的 BERT 列表，可以参考 ``nemo_nlp.huggingface.BERT.list_pretrained_models()`` 。

    .. code-block:: python

        tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
        pretrained_bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name=args.pretrained_bert_model)

如果你在使用一个自己预训练好的 BERT 模型，你可以使用下面这条命令。这里，你需要把 ``args.bert_checkpoint`` 这个参数改成你的 checkpoint 文件所在的位置。

    .. code-block:: python

        tokenizer = SentencePieceTokenizer(model_path=tokenizer_model)
        tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

        bert_model = nemo_nlp.huggingface.BERT(
                config_filename=args.bert_config)
        pretrained_bert_model.restore_from(args.bert_checkpoint)

现在，创建训练和验证数据集合:

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

接着，我们需要在预先训练好的模型上，创建分类器并定义损失函数:

    .. code-block:: python

        hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
        tag_ids = train_dataset.tag_ids
        ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                                  num_classes=len(tag_ids),
                                                  dropout=args.fc_dropout)
        ner_loss = nemo_nlp.TokenClassificationLoss(num_classes=len(tag_ids))

并创建管道用来进行训练和验证:

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

现在，我们设置3个回调函数：

* `SimpleLossLoggerCallback` 打印出训练过程中的损失函数值
* `EvaluatorCallback` 来验证我们dev集合上F1的值。在这个例子中， `EvaluatorCallback` 也会打印出 `output.txt` 上的预测值，这有利于找出模型哪个部分出了问题。
* `CheckpointCallback` 用于保存和读取checkpoints.

.. tip::

    Tensorboard_ 是一个非常好用的调试工具。它在本教程中不是一个必须安装的工具，如果你想使用的话，需要先安装 tensorboardX_ 接着在微调过程中使用如下的命令：

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

最后，我们需要定义学习率规则和优化器，并且开始训练：

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

使用其它的 BERT 模型
--------------------

除了可以使用谷歌提供的预训练 BERT 模型和你自己训练的模型外，在NeMo中，也可以使用来自第三方的BERT模型，只要这个模型的参数可以加载到PyTorch中即可。例如，如果你想使用 SciBERT_ 来微调：

.. _SciBERT: https://github.com/allenai/scibert

.. code-block:: bash

    wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar
    tar -xf scibert_scivocab_cased.tar
    cd scibert_scivocab_cased
    tar -xzf weights.tar.gz
    mv bert_config.json config.json
    cd ..

接着，当你加载你的 BERT 模型，你需要指定模型所在的目录名：

.. code-block:: python

    tokenizer = NemoBertTokenizer(pretrained_model="scibert_scivocab_cased")
    bert_model = nemo_nlp.huggingface.BERT(
        pretrained_model_name="scibert_scivocab_cased",
        factory=neural_factory)

如果你想使用 TensorFlow 训练好的模型，例如 BioBERT ，你需要首先使用 Hugging Face 提供的 `model conversion script`_ 进行模型转换，再在 NeMo 中使用这个模型。

.. _model conversion script: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_tf_checkpoint_to_pytorch.py
