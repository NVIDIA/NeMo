教程
========


简介
-----

**对话状态追踪 (DST)** :cite:`nlp-dst-henderson2015machine` \
的目标是要为正在进行的对话的状态构建一个表示(representation) \
对话是一系列的对话参与者之间的语句。\
换句话说，DST 系统的目标是能捕捉到用户的目标和意图 \
然后把它们编码成一系列的**槽(slots)**和槽对应的**值(values)**。


.. figure:: dst_multiwoz_example.png

   Fig. 1: 一个例子, 多领域对话和相关的状态追踪 (来源: \
   :cite:`nlp-dst-wu2019transferable`)


在这个教程中我们关注多领域对话数据集 MultiWOZ :cite:`nlp-dst-budzianowski2018multiwoz` \
展示如何构建一个 TRADE 模型 :cite:`nlp-dst-wu2019transferable`, \
一个最近发表的领域先进的模型 \
**多领域(Multi-domain)** 场景会引入一些挑战, 最重要的来自于需要 \
**多轮映射(multi-turn mapping)**。在一个 **单轮映射(single-turn mapping)** 场景，(**领域(domain)**, **槽(slot)**, **值(value)**) 三元组可以从 \
单轮中就能推断出。在多轮对话中，这个假设并不存在，DST 系统必须能够从多轮中 \
推断出, 这些信息有可能是横跨多个不同的领域的。




MultiWOZ 数据集
--------------------

多领域数据集 Wizard-of-Oz (`MultiWOZ`_) 是一个囊括了 \
7个领域包含超过10,000个对话的人-人数据集。
原先的 MultiWOZ 2.0 数据集是这篇文章引入的 :cite:`nlp-dst-budzianowski2018multiwoz`.
然而，在这个教程中我们用数据集 MultiWOZ 2.1  :cite:`nlp-dst-eric2019multiwoz`, 它 MultiWOZ 2.0 的升级版。它和原先的数据集有固定的一些问题像是状态的错误，语句错误，值规范化的问题等)。我们的模型也可以在 MultiWOZ 2.0 上训练。

.. _MultiWOZ: https://www.repository.cam.ac.uk/handle/1810/294507

数据集包含下面这些领域:
 1. restaurant
 2. hotel
 3. attraction
 4. taxi
 5. train
 6. hospital
 7. police.

和下面这些槽:
 * inform (∗)
 * address (∗)
 * postcode (∗)
 * phone (∗)
 * name (1234)
 * no of choices (1235)
 * area (123)
 * pricerange (123)
 * type (123)
 * internet (2)
 * parking (2)
 * stars (2)
 * open hours (3)
 * departure (45)
 * destination (45)
 * leave after (45)
 * arrive by (45)
 * no of people (1235)
 * reference no. (1235)
 * trainID (5)
 * ticket price (5)
 * travel time (5)
 * department (7)
 * day (1235)
 * no of days (123).


请注意，一些动作(actions)和槽只和特定领域有关，但一些是全部通用的, \
比如，领域无关的。后者用(∗)表示。


MultiWOZ 数据集有 10,438 个对话，总共 115,434 轮。 \
对话通常分成单和多领域对话。 \
对话长度分布从 1 到 31，大约 70% 对话有超过 10 轮。 \
平均轮数是对单领域和多领域分别为 8.93 和 15.39。 \

每个对话包括一个目标，多个用户和系统语句以及一个信念状态(belief state)和每轮的对话操作(action)以及相应的槽 \
另外，每个对话都有一个任务描述。 \
而且，它包含了系统和用户对话操作(act)的标注 (后者在 MultiWOZ 2.1 中引入).


TRADE 模型
---------------

**TRA**\nsferable **D**\ialogue stat\ **E** generator (TRADE) :cite:`nlp-dst-wu2019transferable` 是为 \
多领域面向任务的对话状态追踪问题
特别设计的模型 \
模型从语句和历史中生成对话状态。它为领域和槽学习嵌入(embeddings)并且 \
受益于拷贝机制(copy mechanism)从而能够促进领域之间的知识转移。它使得模型能够预测 \
在给定领域中，训练过程中从未见过的(**领域(domain)**, **槽(slot)**, **值(value)**)三元组。


.. figure:: dst_trade_architecture.png

   Fig. 2: TRADE 模型的架构 (来源: :cite:`nlp-dst-wu2019transferable`)

模型由三个主要部分组成:

 * 一个 **语句编码器(utterance encoder)**，
 * 一个 **槽栅(slot gate)**，以及
 * 一个 **状态生成器(state generator)**。  

**语句编码器(utterance encoder)** 是一个双向 Gated Recurrent Unit (GRU), 返回上下文单词以及 \
一个编码了整个对话历史的上下文向量。

**状态生成器(state generator)** 也用了 GRU 来预测(domain, slot)对的值。生成器用了一个 soft-gated \
pointer-generator copying，把 **词表上的分布** 和 **对话历史上的分布** 
合成一个单独的输出分布。

最后，**槽栅(slot gate)** 是个简单的分类器，把编码器隐状态的上下文向量 \
映射到三个类上的概率分布: *ptr*, *none*,  和 *dontcare*.

数据预处理
-------------------

首先，你需要从 `MultiWOZ2.1`_ 项目网站上下载 `MULTIWOZ2.1.zip` 。它包含了 \
MultiWOZ 2.1 数据集。或者，你可以从 `MultiWOZ2.0`_ 上下载压缩文件 `MULTIWOZ2.zip` \
它包含了这个数据集的老版本。

.. _MultiWOZ2.1: https://www.repository.cam.ac.uk/handle/1810/294507

.. _MultiWOZ2.0: https://www.repository.cam.ac.uk/handle/1810/280608

接着我们需要预处理，重新格式化我们的数据集，这会将数据集分成三个分布:

 * traininig split ( ``train_dials.json`` 文件包含了8242个对话)
 * validation split ( ``val_dials.json`` 文件包含了1000个对话)
 * test split (``test_dials.json`` 文件包含了999个对话)

你可以用提供好的 `process_multiwoz.py`_ 脚本
预处理 MultiWOZ 数据集:

.. _process_multiwoz.py: https://github.com/NVIDIA/NeMo/tree/master/examples/nlp/dialogue_state_tracking/multiwoz/process_multiwoz.py

.. code-block:: bash

    cd examples/nlp/dialogue_state_tracking/multiwoz
    python process_multiwoz.py

.. note::
    默认情况下，脚本假设你会把数据拷贝以及解压到 \
    ``~/data/state_tracking/multiwoz2.1/`` \
    目录下，并且它会把结果存到 ``~/data/state_tracking/multiwoz2.1`` 文件夹下 \
    你可以在命令行中传入参数 ``source_data_dir`` 和 ``target_data_dir`` \
    来修改。MultiWOZ 2.0 和 MultiWOZ 2.1 可以用相同的脚本处理。


构建 NeMo 图
-----------------------

NeMo 训练图包括六个模块包括数据层，编码器，解码器和损失函数:

 * data_layer (:class:`nemo.collection.nlp.nm.data_layers.MultiWOZDataLayer`)
 * encoder (:class:`nemo.backends.pytorch.common.EncoderRNN`)
 * decoder (:class:`nemo.collection.nlp.nm.trainables.TRADEGenerator`)
 * gate_loss_fn (:class:`nemo.backends.pytorch.common.losses.CrossEntropyLossNM`)
 * ptr_loss_fn (:class:`nemo.collections.nlp.nm.losses.MaskedLogLoss`)
 * total_loss_fn (:class:`nemo.collection.nlp.nm.losses.LossAggregatorNM`)

训练
--------

想要在数据集 MultiWOZ 2.1 上训练 TRADE 模型的实例，并且在它的测试数据集上进行评估，只需要 \
用默认参数运行 `dialogue_state_tracking_trade.py`_ :

.. _dialogue_state_tracking_trade.py: https://github.com/NVIDIA/NeMo/tree/master/examples/nlp/dialogue_state_tracking/dialogue_state_tracking_trade.py


.. code-block:: bash

    cd examples/nlp/dialogue_state_tracking
    python dialogue_state_tracking_trade.py 


.. note::
    同样地，这个脚本会默认读取 ``~/data/state_tracking/multiwoz2.1`` 文件夹.
    这个路径可以用 ``data_dir`` 覆盖。



指标和结果
-------------------

在下面的表格中我们比较了我们实现的 TRADE 模型结果和 \
原始论文 :cite:`nlp-dst-wu2019transferable` 中的结果。在作者们回复 MultiWOZ 2.0
数据集的结果时候, 我们跑了在 MultiWOZ 2.1 数据集上的原始实现，也记录了这些结果。

我们用了和原始实现中相同的参数。在我们的实现和原始的视线中有些区别。\
主要的区别是我们的模型没有用预训练的词嵌入，似乎是会影响模型的效果的。 \
另一个区别是我们在学习策略的时候用了 SquareAnnealing 而不是 \
固定的学习率。另外，我们是根据训练集创建的词表，而原始实现 \
是根据所有数据集包括测试和验证集创建的。我们模型的准确率的主要提升是 \
用了更好的学习率策略。当我们用固定的学习率 \
我们得到了和原始实现中相似的结果。

我们再模型实现上也做了一些提升来加快训练。这使得我们的实现比原始的实现快很多 \
另外, NeMo 支持多 GPU 训练，这使得训练时间更快了。 \
需要注意的是在用多 GPU 的时候学习率应该调高， \
因为 batch size 变大了。

根据 :cite:`nlp-dst-wu2019transferable`, 我们用两个指标来衡量模型的性能:

 * **联合目标准确率(Joint Goal Accuracy)** 比较了每轮对话中的预测对话状态和真实状态，并且输出只有当输出的**所有的值完全正确**
    才会认为输出是正确的。
 * **槽准确率(Slot Accuracy)** 独立地比较每个(domain, slot, value)三元组和它的真实值。


+---------------------------------------------+--------+--------+--------+--------+--------+--------+--------+--------+
|                                             | MultiWOZ 2.0                      | MultiWOZ 2.1                      |
+                                             +--------+--------+--------+--------+--------+--------+--------+--------+
|                                             | Test            |Development      |  Test           |Development      |
+                                             +--------+--------+--------+--------+--------+--------+--------+--------+
| TRADE implementations                       | Goal   | Slot   | Goal   | Slot   | Goal   | Slot   | Goal   | Slot   |
+=============================================+========+========+========+========+========+========+========+========+
| Original :cite:`nlp-dst-wu2019transferable` | 48.62% | 96.92% | 48.76% | 96.95% | 45.31% | 96.57% | 49.15% | 97.04% |
+---------------------------------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| NeMo's Implementation of TRADE              | 48.92% | 97.03% | 50.96% | 97.17% | 47.25% | 96.80% | 51.38% | 97.21% |
+---------------------------------------------+--------+--------+--------+--------+--------+--------+--------+--------+


.. note::
    在训练 TRADE 模型的时候用一个额外的监督信号，强制 Slot Gate 能够恰当的分类 \
    上下文向量。脚本 `process_multiwoz.py`_ 从数据集中抽取额外的信息,
    脚本 `dialogue_state_tracking_trade.py`_ 也汇报了 **Gating Accuracy**。

参考
-------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-DST
    :keyprefix: nlp-dst-
