BERT 预训练
============

在本教程中，我们会按照 BERT 模型结构 :cite:`nlp-bert-devlin2018bert` 构建并训练一个掩码语言模型。训练可以完全从零开始或者在一个预训练好的模型基础上继续训练。在开始本教程之前，请先安装好 ``nemo`` 和 ``nemo_nlp`` 。关于安装 ``nemo`` 的一些步骤可以参阅 :ref:`installation` 章节。

创建一个专门领域的BERT模型对于某些应用是更有优势的。比如一个专门针对生物医学领域的专业 BERT ，类似于 BioBERT :cite:`nlp-bert-lee2019biobert` 和 SciBERT :cite:`nlp-bert-beltagy2019scibert` 。

本教程中所使用的代码来自于 `examples/nlp/language_modeling/bert_pretraining.py`.

.. tip::
    我们提供了几个 BERT 预训练模型，您可以直接使用：
    `bert large uncased for nemo <https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedfornemo>`__
    `bert base uncased for nemo <https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedfornemo>`__
    `bert base cased for nemo <https://ngc.nvidia.com/catalog/models/nvidia:bertbasecasedfornemo>`__

简介
------------

创建领域相关的 BERT 模型对于很多应用都有溢。一个显著的领域相关的 BERT 模型的例子是生物医学的场景下，
比如 BioBERT :cite:`nlp-bert-lee2019biobert` 和 SciBERT :cite:`nlp-bert-beltagy2019scibert`.

.. _bert_data_download:

语料下载
--------

训练语料可以是原始的文本数据，也可以是预处理过的数据。如果是原始文本，我们需要在训练的过程中进行文本处理。接下来，我们会分别说明这两种情况。
首先我们演示如何在原始文本数据上做训练。我们使用一个非常小的英文数据集 WikiText-2 :cite:`nlp-bert-merity2016pointer` 。

运行脚本 `examples/nlp/language_modeling/get_wkt2.sh <data_dir>` 便可以下载这个数据集。下载后并解压，会得到如下三个文件：

.. code-block:: bash

    test.txt
    train.txt
    valid.txt

如果想尝试训练中文 BERT 模型，你可以下载中文维基语料 wiki2019zh_ 。下载后，你需要解压并用这个脚本 `examples/nlp/language_modeling/process_wiki_zh.py` 来进行预处理

.. _wiki2019zh: https://github.com/brightmart/nlp_chinese_corpus

.. code-block:: bash

    python examples/nlp/scripts/process_wiki_zh.py --data_dir=./wiki_zh --output_dir=./wiki_zh --min_frequency=3


你也可以选择已经预处理好的数据进行训练。我们使用 BERT 论文中提及的维基百科和 BookCorpus 数据集。

想要下载数据集，前往 `这个网址 <https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT>`__
然后运行脚本文件 `./data/create_datasets_from_start.sh` 。
如果顺利的话，你会得到两个名字类似于这样 `lower_case_[0,1]_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5`
和 `lower_case_[0,1]_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5` 的文件夹。他们分别包含序列长度为128和512的数据。


创建分词器(Tokenizer)
---------------------

对于原始文本数据，你需要一个分词器来进行处理。
首先你需要创建一个 `BERTPretrainingDataDesc` 对象来描述数据集的格式。这其中涉及的主要步骤包括将数据集符号化并创建词表 (vocabulary) 和一个分词器 (tokenizer) .

你也可以使用一个现成的词表或者分词器模型来跳过这一步。如果你已经有一个预训练好的分词器模型，将它复制到文件夹 `[data_dir]/bert` 下并重命名为 `tokenizer.model` 。

如果你有一个现成的词表文件，可以将它复制到文件夹 `[data_dir]/bert` 下并命名为 `vocab.txt` 。

.. code-block:: python

    import nemo.collections.nlp as nemo_nlp

    data_desc = nemo_nlp.data.BERTPretrainingDataDesc(
                    dataset_name=args.dataset_name,
                    train_data=args.data_dir,
                    vocab_size=args.vocab_size,
                    sample_size=args.sample_size,
                    special_tokens=special_tokens)

接下来我们需要定义tokenizer。如果你想使用一个自定义的词表文件，我们强烈推荐使用 `SentencePieceTokenizer` 。如果要训练中文BERT模型，请使用 `NemoBertTokenizer` 。

.. code-block:: python

    # If you're using a custom vocabulary, create your tokenizer like this
    tokenizer = nemo_nlp.data.SentencePieceTokenizer(model_path="tokenizer.model")
    special_tokens = nemo_nlp.data.get_bert_special_tokens('bert')
    tokenizer.add_special_tokens(special_tokens)

    # Otherwise, create your tokenizer like this
    tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model="bert-base-uncased")
    # or
    tokenizer = nemo_nlp.data.NemoBertTokenizer(vocab_file="vocab.txt")


创建模型
--------

.. tip::

    建议你在一个 Jupyter notebook 中尝试以下内容，以方便调试。

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

如果想从一个已有的模型开始训练，你可以指定选项 `--load_dir` 和类似于下面这样的代码：

.. code-block:: python

    ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                        load_from_folder=args.load_dir)

如果你想从一个已有的 BERT 模型文件继续训练，那设置一个模型的名字即可。如果想查看完整的预训练好的 BERT 模型列表，可以使用 `nemo_nlp.huggingface.BERT.list_pretrained_models()` 。

    .. code-block:: python

        bert_model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name="bert-base-cased")

接下来，我们需要定义分类器和损失函数。在本教程中，我们会同时使用掩码语言模型和预测下一句模型这两个模型的损失函数，如果你只用掩饰语言模型作为损失的话，可能会观察到更高的准确率。

    .. code-block:: python

        mlm_classifier = nemo_nlp.nm.trainables.BertTokenClassifier(
                                    args.hidden_size,
                                    num_classes=args.vocab_size,
                                    activation=ACT2FN[args.hidden_act],
                                    log_softmax=True)

        mlm_loss_fn = nemo_nlp.nm.losses.SmoothedCrossEntropyLoss()

        nsp_classifier = nemo_nlp.nm.trainables.SequenceClassifier(
                                                args.hidden_size,
                                                num_classes=2,
                                                num_layers=2,
                                                activation='tanh',
                                                log_softmax=False)

        nsp_loss_fn = nemo.backends.pytorch.common.CrossEntropyLossNM()

        bert_loss = nemo.backends.pytorch.common.losses.LossAggregatorNM(num_inputs=2)

之后，我们将 encoder embedding 层的权重与 MLM 输出层绑定：

    .. code-block:: python

        mlm_classifier.tie_weights_with(
            bert_model,
            weight_names=["mlp.last_linear_layer.weight"],
            name2name_and_transform={
                "mlp.last_linear_layer.weight": ("bert.embeddings.word_embeddings.weight", nemo.core.WeightShareTransform.SAME)
            },
        )

然后，我们把从输入到输出的整个计算流程封装成一个函数。有了这个函数，我们就可以很方便的分别创建训练计算图和评估计算图。

如果用的是原始文本数据，则选择 `nemo_nlp.nm.data_layers.BertPretrainingDataLayer` 。如果是预处理好的数据，则选择 `nemo_nlp.nm.data_layers.BertPretrainingPreprocessedDataLayer`

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
        #        batch_size,
        #        mode)

        steps_per_epoch = len(data_layer) // (batch_size * args.num_gpus * args.batches_per_step)

        input_data = data_layer()

        hidden_states = bert_model(input_ids=input_data.input_ids,
                                   token_type_ids=input_data.input_type_ids,
                                   attention_mask=input_data.input_mask)

        mlm_logits = mlm_classifier(hidden_states=hidden_states)
        mlm_loss = mlm_loss_fn(logits=mlm_logits,
                               labels=input_data.output_ids,
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
                                batches_per_step=args.batches_per_step,
                                mode="train")

    # for preprocessed data 
    # train_loss, _, _, steps_per_epoch = create_pipeline(
    #                            data_file=args.train_data,
    #                            preprocessed_data=True,
    #                            max_predictions_per_seq=args.max_predictions_per_seq,
    #                            batch_size=args.batch_size,
    #                            batches_per_step=args.batches_per_step,
    #                            mode="train")

    eval_loss, _, _, _ = create_pipeline(
                                    data_file=data_desc.eval_file,
                                    preprocessed_data=False,
                                    max_seq_length=args.max_seq_length,
                                    mask_probability=args.mask_probability,
                                    short_seq_prob=args.short_seq_prob,
                                    batch_size=args.batch_size,
                                    batches_per_step=args.batches_per_step,
                                    mode="eval")

    # for preprocessed data 
    # eval_loss, eval_mlm_loss, eval_nsp_loss, _ = create_pipeline(
    #                            data_file=args.eval_data,
    #                            preprocessed_data=True,
    #                            max_predictions_per_seq=args.max_predictions_per_seq,
    #                            batch_size=args.batch_size,
    #                            batches_per_step=args.batches_per_step,
    #                            mode="eval")


运行
----

接着定义学习率：

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                    total_steps=args.num_iters,
                                    warmup_ratio=args.lr_warmup_proportion)

        # if you are training on raw text data, you have use the alternative to set the number of training epochs
        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)

再然后，我们定义一些必要的回调函数：

1. `SimpleLossLoggerCallback`: 跟踪训练过程中损失函数的变化
2. `EvaluatorCallback`: 跟踪评估集上的指标变化
3. `CheckpointCallback`: 每过一段时间间隔保存模型

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(tensors=[train_loss],
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())))),
            step_freq=args.train_step_freq,
        eval_callback = nemo.core.EvaluatorCallback(eval_tensors=[eval_loss],
            user_iter_callback=nemo_nlp.callbacks.lm_bert_callback.eval_iter_callback,
            user_epochs_done_callback=nemo_nlp.callbacks.lm_bert_callback.eval_epochs_done_callback
            eval_step=args.eval_step_freq)
        ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
            epoch_freq=args.save_epoch_freq,
            load_from_folder=args.load_dir,
            step_freq=args.save_step_freq)


我们还建议把模型参数保存到一个配置文件中。这样做的话，你以后使用 NeMo 的时候导入 BERT 模型会非常方便。

    .. code-block:: python

        config_path = f'{nf.checkpoint_dir}/bert-config.json'

        if not os.path.exists(config_path):
            bert_model.config.to_json_file(config_path)

最后，我们定义优化器并开始训练！

    .. code-block:: python

        nf.train(tensors_to_optimize=[train_loss],
                 lr_policy=lr_policy_fn,
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 optimizer=args.optimizer,
                 optimization_params={"batch_size": args.batch_size,
                                      "num_epochs": args.num_epochs,
                                      "lr": args.lr,
                                      "betas": (args.beta1, args.beta2),
                                      "weight_decay": args.weight_decay})

如何使用样例中的训练脚本
------------------------

完整的 BERT 模型训练脚本保存在这个文件中： `examples/nlp/language_modeling/bert_pretraining.py`

如果想进行单个 GPU 的训练，可以运行这个命令：

.. code-block:: bash

    cd examples/nlp/language_modeling
    python bert_pretraining.py [args]


如果想进行多 GPU 训练，可以运行：

.. code-block:: bash

    cd examples/nlp/language_modeling
    python -m torch.distributed.launch --nproc_per_node=x bert_pretraining.py --num_gpus=x [args]

如果使用的是原始的文本数据，请在命令行中添加选项 ``data_text``

.. code-block:: bash

    python bert_pretraining.py [args] data_text [args]

如果使用的是预处理过的数据（默认配置），请使用 ``data_preprocessed``

.. code-block:: bash

    python bert_pretraining.py [args] data_preprocessed [args]

.. note::
    关于下载和预处理数据，请参阅 :ref:`bert_data_download`

.. tip::

    Tensorboard_ 是一个非常棒的调试工具。虽然不是训练的必要步骤，但是你可以安装 tensorboardX_ 并在训练过程中运行它来观察一些指标在训练过程中的变化：

    .. code-block:: bash

        tensorboard --logdir bert_pretraining_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX


参考
----

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-BERT-PRETRAINING
    :keyprefix: nlp-bert-
