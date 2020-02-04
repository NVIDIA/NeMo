教程
====

在教程前，请确认你已经安装了 ``nemo`` 和 ``nemo_nlp`` 。你可以通过这个部分获得更多的信息 :ref:`installation`

简介
----

这个教程将介绍如何在NeMo中，实现命名实体识别(Named Entity Recognition, NER)。我们将通过一个预训练好的 BERT 模型来进行展示，或者你也可以使用一个训练好的模型！你可以通过 BERT 预训练教程获得更多的信息。

下载数据集
----------

`CoNLL-2003`_ 是一个NER上标准的验证集合，当然任何一个NER数据集合都可以。数据集合被拆分成两个文件 text.txt 和 labels.txt。text.txt 需要满足的格式为：

.. _CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/

.. code-block:: bash

    Jennifer is from New York City .
    She likes ...
    ...

labels.txt 需要满足的格式为：

.. code-block:: bash

    B-PER O O B-LOC I-LOC I-LOC O
    O O ...
    ...

text.txt 每一行包含文本序列，其中词以空格来进行分隔。label.txt 中包含 text.txt 中每个词的标注，标注以空格分隔。文件中的每一行需要符合如下格式：
[WORD] [SPACE] [WORD] [SPACE] [WORD] (在 text.txt 中) 和 [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (在 labels.txt中)。

.. _script: https://github.com/NVIDIA/NeMo/tree/master/scripts/get_conll_data.py


训练
----

.. tip::

    我们建议试试使用Jupyter来运行这部分代码，这会使得调试更加容易!
    详见 examples/nlp/NERWithBERT.ipynb

首先，我们需要使用所支持的后端，来创建我们的 `Neural Factory` 。你需要确认使用多GPU或者混合精度训练。这个教程中我们使用单GPU训练，不使用混合精度(``optimization_level="O0"``)。如果你想使用混合精度训练，需要设置 ``amp_opt_level`` 这个参数为 ``O1`` 或者 ``O2`` 。

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

        tokenizer = NemoBertTokenizer(pretrained_model="bert-base-cased")
        bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-cased")

查看 examples/nlp/token_classification.py 文件来获取如何使用自己预训练好的模型。
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

        label_ids = train_data_layer.dataset.label_ids
        num_classes = len(label_ids)

        hidden_size = bert_model.hidden_size
        ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                              num_classes=num_classes,
                                              dropout=CLASSIFICATION_DROPOUT)

        ner_loss = nemo_nlp.TokenClassificationLoss(d_model=hidden_size,
                                                num_classes=len(label_ids),
                                                dropout=CLASSIFICATION_DROPOUT)

现在，创建训练和验证集合:

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

    .. code-block:: python

        train_tensors, train_loss, steps_per_epoch, label_ids, _ = create_pipeline()
        eval_tensors, _, _, _, data_layer = create_pipeline(mode='dev')

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

        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss],
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())))

        train_data_size = len(train_data_layer)

        # 如果你在使用多 GPU 训练，需要把这里改成
        # train_data_size / (batch_size * batches_per_step * num_gpus)
        steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_logits, eval_labels, eval_subtokens_mask],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, label_ids),
            eval_step=steps_per_epoch)

最后，我们需要定义学习率规则和优化器，并且开始训练：

    .. code-block:: python

        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=[train_callback, eval_callback],
                 lr_policy=lr_policy,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

使用脚本训练新的 BERT 模型
--------------------------

运行如下训练脚本:

.. code-block:: bash

    python token_classification.py --data_dir /data/ner/ --work_dir output_ner

测试:

.. code-block:: bash

    python token_classification_infer.py --labels_dict /data/ner/label_ids.csv
    --work_dir output_ner/checkpoints/

注意，这里会在训练过程中，到 data_dir 目录下生成 label_ids.csv 文件。

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
        pretrained_model_name="scibert_scivocab_cased"
    )

如果你想使用 TensorFlow 训练好的模型，例如 BioBERT ，你需要首先使用 Hugging Face 提供的 `model conversion script`_ 进行模型转换，再在 NeMo 中使用这个模型。

.. _model conversion script: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_tf_checkpoint_to_pytorch.py
