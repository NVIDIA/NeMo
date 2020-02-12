BERT预训练
==========

在本教程中，我们会按照BERT模型结构 :cite:`nlp-bert-devlin2018bert` 构建并训练一个掩码语言模型。训练可以完全从零开始或者在一个预训练好的模型基础上继续训练。在开始本教程之前，请先安装好 ``nemo`` 和 ``nemo_nlp`` 。关于安装 ``nemo`` 的一些步骤可以参阅 :ref:`installation` 章节。

创建一个专门领域的BERT模型对于某些应用是更有优势的。比如一个专门针对生物医学领域的专业BERT，类似于BioBERT :cite:`nlp-bert-lee2019biobert` 和SciBERT :cite:`nlp-bert-beltagy2019scibert` 。

本教程中所使用的代码来自于 ``examples/nlp/language_modeling/bert_pretraining.py``.

语料下载
--------

因为这只是一个演示，所以我们使用一个非常小的英文数据集 WikiText-2 :cite:`nlp-bert-merity2016pointer` 。

运行脚本 ``examples/nlp/scripts/get_wt2.sh`` 便可以下载这个数据集。下载后并解压，会得到如下三个文件：

    .. code-block:: bash

        test.txt
        train.txt
        valid.txt

如果想尝试训练中文BERT模型，你可以下载中文维基语料 wiki2019zh_。下载后，你需要解压并用这个脚本 ``examples/nlp/scripts/process_wiki_zh.py`` 来进行预处理

.. _wiki2019zh: https://github.com/brightmart/nlp_chinese_corpus

    .. code-block:: bash

        python examples/nlp/scripts/process_wiki_zh.py --data_dir=./wiki_zh --output_dir=./wiki_zh --min_frequency=3

创建分词器(Tokenizer)
---------------------
首先你需要创建一个 `BERTPretrainingDataDesc` 对象来描述数据集的格式。这其中涉及的主要步骤包括将数据集符号化并创建词表(vocabulary)和一个分词器(tokenizer).

你也可以使用一个现成的词表或者分词器模型来跳过这一步。如果你已经有一个预训练好的分词器模型，将它复制到文件夹 ``[data_dir]/bert`` 下并重命名为 ``tokenizer.model`` 。

如果你有一个现成的词表文件，可以将它复制到文件夹 ``[data_dir]/bert`` 下并命名为 ``vocab.txt`` 。

    .. code-block:: python

        data_desc = BERTPretrainingDataDesc(args.dataset_name,
                                            args.data_dir,
                                            args.vocab_size,
                                            args.sample_size,
                                            special_tokens,
                                            'train.txt')

接下来我们需要定义tokenizer。如果你想使用一个自定义的词表文件，我们强烈推荐使用 `SentencePieceTokenizer` 。如果要训练中文BERT模型，请使用 `NemoBertTokenizer` 。

    .. code-block:: python

        # If you're using a custom vocabulary, create your tokenizer like this
        tokenizer = SentencePieceTokenizer(model_path="tokenizer.model")
        special_tokens = {
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "bos_token": "[CLS]",
            "mask_token": "[MASK]",
            "eos_token": "[SEP]",
            "cls_token": "[CLS]",
        }
        tokenizer.add_special_tokens(special_tokens)

        # Otherwise, create your tokenizer like this
        tokenizer = NemoBertTokenizer(vocab_file="vocab.txt")
        # or
        tokenizer = NemoBertTokenizer(pretrained_model="bert-base-uncased") 

创建模型
--------

.. tip::

    建议你在一个Jupyter notebook中尝试以下内容，以方便调试。

首先，我们需要创建一个 `NeuralModuleFactory` 对象并调用所支持的后端。具体如何创建还取决于你是否想进行多GPU训练或者混合精度训练等。在本教程中，我们只使用一个GPU，而且没有混合精度训练。如果你想使用混合精度训练，可以将 ``amp_opt_level`` 选项设置为 ``O1`` 或者 ``O2`` 。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

接下来我们需要定义模型结构。这里我们从 `huggingface` 模块导入BERT的模型结构。你只需要定义一些关键参数即可。

    .. code-block:: python

        bert_model = nemo_nlp.nm.trainables.huggingface.BERT(
            vocab_size=args.vocab_size,
            num_hidden_layers=args.num_hidden_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_seq_length,
            hidden_act=args.hidden_act)

如果你想从一个已有的BERT模型文件继续训练，那设置一个模型的名字即可。如果想查看完整的预训练好的BERT模型列表，可以使用 `nemo_nlp.huggingface.BERT.list_pretrained_models()` 。

    .. code-block:: python

        bert_model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name="bert-base-cased")

接下来，我们需要定义分类器和损失函数。在本教程中，我们会同时使用掩码语言模型和预测下一句模型这两个模型的损失函数，如果你只用掩饰语言模型作为损失的话，可能会观察到更高的准确率。

    .. code-block:: python

        mlm_classifier = nemo_nlp.nm.trainables.TokenClassifier(args.d_model,
                                                  num_classes=tokenizer.vocab_size,
                                                  num_layers=1,
                                                  log_softmax=True)
        mlm_loss_fn = nemo_nlp.nm.losses.MaskedLanguageModelingLossNM()

        nsp_classifier = nemo_nlp.nm.trainables.SequenceClassifier(args.d_model,
                                                     num_classes=2,
                                                     num_layers=2,
                                                     log_softmax=True)
        nsp_loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss()

        bert_loss = nemo_nlp.nm.losses.LossAggregatorNM(num_inputs=2)

然后，我们把从输入到输出的整个计算流程封装成一个函数。有了这个函数，我们就可以很方便的分别创建训练流和评估流：

    .. code-block:: python

        def create_pipeline(**args):
                    data_layer = nemo_nlp.nm.data_layers.BertPretrainingDataLayer(
                                            tokenizer,
                                            data_file,
                                            max_seq_length,
                                            mask_probability,
                                            short_seq_prob,
                                            batch_size)
                    # for preprocessed data
                    # data_layer = nemo_nlp.BertPretrainingPreprocessedDataLayer(
                    #        data_file,
                    #        max_predictions_per_seq,
                    #        batch_size, is_training)

                    steps_per_epoch = len(data_layer) // (batch_size * args.num_gpus * args.batches_per_step)

                    input_data = data_layer()

                    hidden_states = bert_model(input_ids=input_data.input_ids,
                                            token_type_ids=input_data.input_type_ids,
                                            attention_mask=input_data.input_mask)

                    mlm_logits = mlm_classifier(hidden_states=hidden_states)
                    mlm_loss = mlm_loss_fn(logits=mlm_logits,
                                        output_ids=input_data.output_ids,
                                        output_mask=input_data.output_mask)

                    nsp_logits = nsp_classifier(hidden_states=hidden_states)
                    nsp_loss = nsp_loss_fn(logits=nsp_logits, labels=input_data.labels)

                    loss = bert_loss(loss_1=mlm_loss, loss_2=nsp_loss)

                    return loss, mlm_loss, nsp_loss, steps_per_epoch


                train_loss, _, _, steps_per_epoch = create_pipeline(
                                            data_file=data_desc.train_file,
                                            preprocessed_data=False,
                                            max_seq_length=args.max_seq_length,
                                            mask_probability=args.mask_probability,
                                            short_seq_prob=args.short_seq_prob,
                                            batch_size=args.batch_size,
                                            batches_per_step=args.batches_per_step)

                # for preprocessed data 
                # train_loss, _, _, steps_per_epoch = create_pipeline(
                #                            data_file=args.data_dir,
                #                            preprocessed_data=True,
                #                            max_predictions_per_seq=args.max_predictions_per_seq,
                #                            training=True,
                #                            batch_size=args.batch_size,
                #                            batches_per_step=args.batches_per_step)

                eval_loss, eval_tensors, _ = create_pipeline(data_desc.eval_file,
                                                            args.max_seq_length,
                                            



再然后，我们定义一些必要的回调函数：

1. `SimpleLossLoggerCallback`: 跟踪训练过程中损失函数的变化
2. `EvaluatorCallback`: 跟踪评估集上的指标变化
3. `CheckpointCallback`: 每过一段时间间隔保存模型

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(...)
        eval_callback = nemo.core.EvaluatorCallback(...)
        ckpt_callback = nemo.core.CheckpointCallback(...)

.. tip::

    Tensorboard_ 是一个非常棒的调试工具。虽然不是训练的必要步骤，但是你可以安装 tensorboardX_ 并在训练过程中运行它来观察一些指标在训练过程中的变化：

    .. code-block:: bash

        tensorboard --logdir bert_pretraining_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX


我们还建议把模型参数保存到一个配置文件中。这样做的话，你以后使用NeMo的时候导入BERT模型会非常方便。

    .. code-block:: python

        config_path = f'{nf.checkpoint_dir}/bert-config.json'

        if not os.path.exists(config_path):
            bert_model.config.to_json_file(config_path)

最后，我们定义优化器并开始训练！

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)

        nf.train(tensors_to_optimize=[train_loss],
                 lr_policy=lr_policy_fn,
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 optimizer=args.optimizer,
                 optimization_params={"batch_size": args.batch_size,
                                      "num_epochs": args.num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay})

参考
----

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-BERT-PRETRAINING
    :keyprefix: nlp-bert-
