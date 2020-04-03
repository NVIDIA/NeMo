教程
====

在本教程中我们将要实现基于 `Transformer 编码器-解码器结构 <https://arxiv.org/abs/1706.03762>`_ :cite:`nlp-nmt-vaswani2017attention` 的神经机器翻译系统。本教程中使用到的所有代码都基于 ``examples/nlp/neural_machine_translation/machine_translation_tutorial.py`` 。

预备知识
--------

**数据集.** 我们使用 WMT16 英文-德文数据集，在数据预处理前，其中包含了约 450 万个句子对。为了获得更加干净的数据，在对句子进行分词并得到符号串（tokens）后，我们从中移除满足以下条件的句子对：

    * 源语言或目标语言句子中符号串（tokens）的数量小于 3 个或多于 128 个的句子对。
    * 源语言和目标语言的绝对差异（Absolute difference）大于 25 个符号串(tokens)的句子对。
    * 源语言（目标语言）的句子长度比目标语言（源语言）大于 2.5 倍的句子对。
    * 目标语言句子与源语言句子完全相同的句子对。

我们使用 newstest2013 数据集作为验证集，并使用 newstest2014 数据集作为测试集。所有的数据集以及分词器（tokenizer）模型都可以从 `此处 <https://drive.google.com/open?id=1AErD1hEg16Yt28a-IGflZnwGTg9O27DT>`_ 下载。 在下面的步骤中，我们假设所有的数据都放置在 **<path_to_data>** 目录中。

**资源.** 本教程中使用的训练脚本 ``examples/nlp/neural_machine_translation/machine_translation_tutorial.py`` 能够训练 Transformer-big 结构的 BERT 模型并在 newstest2014 数据集上达到 **29.2** BLEU / **28.5** SacreBLEU 的分数表现，在配备了多块 16GB Volta 架构图形处理器 的 NVIDIA's DGX-1 上仅需约 15 小时即可完成全部训练过程。同样的训练结果也能够使用更少的资源并通过增加梯度更新的次数来实现 :cite:`nlp-nmt-ott2018scaling` 。

.. tip::
    在不指定任何训练参数的前提下运行训练脚本将会在一个很小的数据集（newstest2013）上开始训练，其中训练集包含 3000 个句子对，验证集包含 100 个句子对。这样训练能够更方便地对代码进行调试：如果一切设置正确，验证集的 BLEU 将很快就能 >99，验证集的损失（loss）也能很快就会 < 1.5。

代码概述
--------
首先，我们实例化一个神经模块工厂对象（Neural Module Factory），这将会定义 1) 后端（backend），2) 混合精度优化级别（mixed precision optimization level），以及 3) 本地 GPU 的编号（local rank）

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=args.work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

然后，我们定义分词器（tokenizer）来将输入文字转化为符号串（tokens）。在本教程中，我们使用在 WMT16 英文-德文语料集上训练的字节对编码 `Byte Pair Encodings (BPE) <https://arxiv.org/abs/1508.07909>`_ :cite:`nlp-nmt-sennrich2015neural` YouTokenToMe 分词器，训练框架使用 `YouTokenToMe 库 <https://github.com/VKCOM/YouTokenToMe>`_ 。与文献中不同的是，文献中通常使用较大的词汇表（30000+ 词汇），而我们使用比文献中小 4 倍的词汇变，其中包含 8192 个字节对（BPEs）。 这样做不仅能达到与使用大词汇相同等级的性能表现，同时也让我们能够在训练时将 batch size 增加 20%，使得模型能够更快地收敛。

    .. code-block:: python

        tokenizer = nemo_nlp.YouTokenToMeTokenizer(
            model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
        vocab_size = 8 * math.ceil(src_tokenizer.vocab_size / 8)


    .. tip::
        为了达到最优的 GPU 利用和混合精度加速，请确保词汇表的大小（以及模型中所有参数的大小）都能够被 8 整除。

如果源语言和目标语言相差较大，则应对源语言和目标语言使用不同的分词器（tokenizer）。例如，当源语言为英文，目标语言为中文，则源语言可以使用 YouTokenToMeTokenizer，目标语言可以使用 CharTokenizer。 这也就意味着模型的输入为英文的字节对编码（BPE），模型的输出为中文的汉字。

    .. code-block:: python

        src_tokenizer = nemo_nlp.YouTokenToMeTokenizer(
            model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
        tgt_tokenizer = nemo_nlp.CharTokenizer(
            vocab_path=f"{args.data_dir}/{args.tgt_tokenizer_model}")

    .. tip::
        使用 CharTokenizer 时应在其构造函数的参数传入词汇表文件（vocab.txt）路径，词汇表文件中应包含对应语言数据中全部的字符。

接下来，我们定义模型中使用到的所有必要的神经模块：

    * Transformer 编码器和解码器。
    * 用于将解码器输出映射到输出词汇概率分布上的 `TokenClassifier` 。
    * 用于生成翻译结果的束搜索（Beam Search）模块。
    * 损失函数：引入标签平滑正则化（label smoothing regularization）的交叉熵（cross entropy）。

    .. code-block:: python

        encoder = nemo_nlp.TransformerEncoderNM(**encoder_params)
        decoder = nemo_nlp.TransformerDecoderNM(**decoder_params)
        log_softmax = nemo_nlp.TokenClassifier(**token_classifier_params)
        beam_search = nemo_nlp.BeamSearchTranslatorNM(**beam_search_params)
        loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(**loss_params)

根据文献 `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`nlp-nmt-press2016using` ，我们将嵌入层（embedding）和分类层（softmax）的参数绑定：

    .. code-block:: python

        log_softmax.log_softmax.dense.weight = encoder.embedding_layer.token_embedding.weight
        decoder.embedding_layer.token_embedding.weight = encoder.embedding_layer.token_embedding.weight

    .. note::
        如果源语言和目标语言使用不同的分词器（tokenizer），请勿进行参数绑定。

然后，我们定义一个将输入转化为输出的管道（pipeline），它将在训练和验证的过程中用到。其中一个重要的部分是数据层（data layer），数据层能够将拥有相似长度的句子封装成批次以最小化填充符号（padding symbol）的使用。

    .. code-block:: python

        def create_pipeline(**args):
            dataset = nemo_nlp.TranslationDataset(**translation_dataset_params)
            data_layer = nemo_nlp.TranslationDataLayer(dataset)
            src, src_mask, tgt, tgt_mask, labels, sent_ids = data_layer()
            src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
            tgt_hiddens = decoder(input_ids_tgt=tgt,
                                  hidden_states_src=src_hiddens,
                                  input_mask_src=src_mask,
                                  input_mask_tgt=tgt_mask)
            logits = log_softmax(hidden_states=tgt_hiddens)
            loss = loss_fn(logits=logits, target_ids=labels)
            beam_results = None
            if not training:
                beam_results = beam_search(hidden_states_src=src_hiddens,
                                           input_mask_src=src_mask)
            return loss, [tgt, loss, beam_results, sent_ids]


        train_loss, _ = create_pipeline(train_dataset_src,
                                        train_dataset_tgt,
                                        args.batch_size,
                                        clean=True)

        eval_loss, eval_tensors = create_pipeline(eval_dataset_src,
                                                  eval_dataset_tgt,
                                                  args.eval_batch_size,
                                                  clean=True,
                                                  training=False)

然后，我们定义必要的回调函数：

1. `SimpleLossLoggerCallback`: 用于追踪训练过程中的损失值
2. `EvaluatorCallback`: 用于追踪在指定间隔时验证数据及上的 BLEU 分数
3. `CheckpointCallback`: 用于保存模型的检查点（checkpoints）

    .. code-block:: python

        from nemo.collections.nlp.callbacks.translation import eval_iter_callback, eval_epochs_done_callback

        train_callback = nemo.core.SimpleLossLoggerCallback(...)
        eval_callback = nemo.core.EvaluatorCallback(...)
        ckpt_callback = nemo.core.CheckpointCallback(...)

    .. note::
        BLEU 分数是通过计算模型预测得到的翻译句子与验证集中真实的目标句子得到的。考虑到完整性，我们计算了两个在文献中常用的指标，分别是 `SacreBLEU <https://github.com/mjpost/sacreBLEU>`_ :cite:`nlp-nmt-post2018call` 和 `tokenized BLEU score <https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl>`_ 。

最后，我们定义优化器的参数并开始训练。

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.max_steps,
                                     warmup_steps=args.warmup_steps)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=callbacks,
                 optimizer=args.optimizer,
                 lr_policy=lr_policy_fn,
                 optimization_params={"num_epochs": max_num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay,
                                      "betas": (args.beta1, args.beta2)},
                 batches_per_step=args.iter_per_step)


模型训练
--------

要想训练一个 Transformer-big 结构的神经机器翻译模型，请运行位于 ``examples/nlp/neural_machine_translation/machine_translation_tutorial.py`` 的 ``nmt_tutorial.py`` ：

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=<num_gpus> nmt_tutorial.py \
            --data_root <path_to_data> --src_tokenizer_model bpe8k_yttm.model \
            --eval_datasets valid/newstest2013 --optimizer novograd --lr 0.04 \
            --weight_decay 0.0001 --max_steps 40000 --warmup_steps 4000 \
            --d_model 1024 --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
            --batch_size 12288 --iter_per_step 5

    .. note::
        这个命令会在 8 块 GPU 上开始模型训练，显存需求最少为 16GB。如果你的 GPU 显存较少，请适量调低 ``batch_size`` 参数，并适量调高 ``iter_per_step`` 参数。

要想训练一个英文-中文的神经机器翻译模型，需要指定 ``--src_lang`` 为 ``en`` ， ``--tgt_lang`` 为 ``zh`` ，同时将 ``--tgt_tokenizer_model`` 设置为词汇表文件的路径，中文训练数据的样例格式请参考 ``/tests/data/nmt_en_zh_sample_data/`` 。

使用预训练的模型进行翻译
------------------------

1. 将你训练时保存的模型检查点（checkpoint）文件（或者可以直接从 `这里 <https://ngc.nvidia.com/catalog/models/nvidia:transformer_big_en_de_8k>`_ 下载检查点文件，该检查点在 newstest2014 数据集上取得了 28.5 的 SacreBLEU 分数）放置到 ``<path_to_ckpt>`` 目录中。
2. 在交互式模式中运行 ``nmt_tutorial.py``::

    python nmt_tutorial.py --src_tokenizer_model bpe8k_yttm.model \
         --eval_datasets test --optimizer novograd --d_model 1024 \
         --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
         --checkpoint_dir <path_to_ckpt> --interactive

   .. image:: interactive_translation.png
       :align: center

引用
----

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-NMT
    :keyprefix: nlp-nmt-
