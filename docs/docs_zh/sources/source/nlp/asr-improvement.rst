教程
====

在这个教程中，我们会训练一个后处理模型来纠正端到端语音识别模型的输出错误。这个模型类似于一个翻译模型，但和传统语音识别中的二次打分模型不太一样。
这个模型的架构是基于注意力机制的编码器解码器架构，其中编码器和解码器都是用BERT的预训练语言模型初始化的。为了训练这个模型，我们用预训练的 Jasper 语音识别模型 :cite:`asr-imps-li2019jasper` 产生的错误来收集数据集。

数据
----
**数据收集** 我们用 Jasper :cite:`asr-imps-li2019jasper` 在 Librispeech 数据集 :cite:`asr-imps-panayotov2015librispeech`  上训练的模型为这个任务收集数据集。
下载 Librispeech 数据集, 参考 :ref:`LibriSpeech_dataset` 。
获得 Jasper 预训练模型, 参考 :ref:`Jasper_model` 。
Librispeech 训练数据集包含三个部分: train-clean-100, train-clean-360, 和 train-clean-500 总共281000个训练样本。
我们用两个方法来扩增数据集：

* 我们把所有的训练集分成10份，然后用交叉验证的方法训练10个 Jasper 模型: 一个模型在9份数据集上训练，然后在剩下的那份数据集上做语音识别。
* 我们用预训练的 Jasper 模型，在训练集上做推理的时候，开启 dropout。这个过程用不同的随机种子重复多次。

**数据后处理** 收集到的数据集需要去除重复以及错词率大于0.5的样本。
得到的数据集包含1,700,000对 "坏" 英文-"好" 英文样本对。

**开发和测试集准备** Librispeech 包含两个开发集 (dev-clean 和 dev-other) 以及2个测试集 (test-clean 和 test-other)。
在我们的任务中，我们也这么分。我们把这些数据集放到预训练好的 Jasper 模型中，用贪婪算法 (greedy) 解码得到语音识别的输出结果。
这些结果在我们的教程中用来做评测。

从预训练 BERT 模型中加载参数
----------------------------
编码器和解码器用的都是预训练的 BERT 模型参数。 因为 BERT 的语言模型和 Transformer 的编码器结构相同，因此没有其他什么需要做的。从预训练的 BERT 模型中为解码器准备参数，我们写了一个脚本 ``get_decoder_params_from_bert.py`` 会从 ``transformers`` 仓库 :cite:`asr-imps-huggingface2019transformers` 下载参数，并把他们映射到解码器的参数上。
编码器和解码器的注意力是用 self-attention 参数做初始化的。
这个脚本位于 ``examples/nlp/asr_postprocessor/get_decoder_params_from_bert.py`` 文件目录下，接受两个参数：

* ``--model_name``: 模型名称，可选择 ``bert-base-cased``, ``bert-base-uncased`` 等参数。
* ``--save_to``: 指定保存目录

    .. code-block:: bash

        $ python get_decoder_params_from_bert.py --model_name bert-base-uncased --save_to results_dir

神经模块概览
------------

首先，因为所有的模块都是由NeMo构建的，我们需要初始化 ``NeuralModuleFactory`` ，我们需要定义 1) 后端(backend) (PyTorch)，2) 混精度优化等级，3) GPU的loca rank以及，4) 一个实验管理器，创建一个时间戳的文件夹来存储 checkpoints 和相关的输出，日志文件以及 TensorBoard 的图。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(
                        backend=nemo.core.Backend.PyTorch,
                        local_rank=args.local_rank,
                        optimization_level=args.amp_opt_level,
                        log_dir=work_dir,
                        create_tb_writer=True,
                        files_to_copy=[__file__])

接着我们定义分词器(tokenizer)，把所有的词转到它们对应的序号上。我们会使用 ``bert-base-uncased`` 模型的词表，因为我们的数据集只包含不区分大小写的文本：

    .. code-block:: python

        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model="bert-base-uncased")

编码器模块对应于 BERT 的语言模型，它来自于 ``nemo_nlp.nm.trainables.huggingface`` 模块：

    .. code-block:: python

        zeros_transform = nemo.backends.pytorch.common.ZerosLikeNM()
        encoder = nemo_nlp.nm.trainables.huggingface.BERT(
            pretrained_model_name=args.pretrained_model,
            local_rank=args.local_rank)


    .. tip::

        让词嵌入的大小（包括其他的张量维度）能够整除8可以得到最好的GPU利用率和混精度训练加速。


    .. code-block:: python

        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
        tokens_to_add = vocab_size - tokenizer.vocab_size

        device = encoder.bert.embeddings.word_embeddings.weight.get_device()
        zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)

        encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
            (encoder.bert.embeddings.word_embeddings.weight.data, zeros))


接着, 我们构建 Transformer 解码器神经模块. 因为我们会用 BERT 预训练的参数来初始化我们的解码器, 我们设置隐藏层激活函数为 ``"hidden_act": "gelu"`` 以及设置学习位置编码 ``"learn_positional_encodings": True`` :

    .. code-block:: python

        decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
            d_model=args.d_model,
            d_inner=args.d_inner,
            num_layers=args.num_layers,
            num_attn_heads=args.num_heads,
            ffn_dropout=args.ffn_dropout,
            vocab_size=vocab_size,
            max_seq_length=args.max_seq_length,
            embedding_dropout=args.embedding_dropout,
            learn_positional_encodings=True,
            hidden_act="gelu",
            **dec_first_sublayer_params)

为了把预训练参数加载到解码器参数中, 我们用解码器神经模块的属性函数 ``restore_from`` 来加载:

    .. code-block:: python

        decoder.restore_from(args.restore_from, local_rank=args.local_rank)

模型训练
--------

训练模型，运行 ``asr_postprocessor.py.py`` ，它位于 ``examples/nlp/asr_postprocessor`` 目录中。我们用 novograd 优化器来训练 :cite:`asr-imps-ginsburg2019stochastic`, 设置学习率 ``lr=0.001`` ，多项式学习率衰减策略, ``1000`` 步预热, 每个GPU的 batch size 为 ``4096*8`` 个符号, 以及 ``0.25`` dropout 概率。我们在8块GPU上做训练，可以用下面的方法开启多GPU训练模式:

    .. code-block:: bash

        $ python -m torch.distributed.launch --nproc_per_node=8  asr_postprocessor.py --data_dir data_dir --restore_from bert-base-uncased_decoder.pt

参考
----

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: ASR-IMPROVEMENTS
    :keyprefix: asr-imps- 
