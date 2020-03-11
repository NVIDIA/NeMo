教程
========


ASR系统通常产生的文本是没有标点符号和不区分词的大小写。这个教程讲述了如果在 NeMo 中实现模型预测标点和为每个词预测是否要首字母大写，从而使得 ASR 的输出更加可读，并且提升下游的任务像是命名实体识别和机器翻译。我们会展示如何用一个预训练的 BERT 模型来训练这个网络。 

.. tip::

    我们建议你在 Jupyter notebook 中尝试这个例子，它位于 examples/nlp/token_classification/PunctuationWithBERT.ipynb.
    
    这个教程中的所有代码都基于 :ref:`punct_scripts`.
    在 NeMo 中预训练 BERT 以及预训练好的模型 checkpoints 请参考 `BERT 预训练 <https://nvidia.github.io/NeMo/zh/nlp/bert_pretraining.html>`__.


任务描述
----------------

对训练集中每个字我们要预测:

1. 跟着这个词的标点符号和
2. 这个词是否要首字母大写

在这个模型中， 我们在预训练的 BERT 模型上联合训练 2 个 token 层面的分类器: 一个预测标点符号，另一个预测大小写。

数据集
-------

模型可以运行在任何数据集上，只要它遵守下面的格式。这个教程中我们会用数据集 `Tatoeba collection of sentences`_. `This`_ 脚本下载和预处理数据集。

.. _Tatoeba collection of sentences: https://tatoeba.org/eng
.. _This: https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/token_classification/get_tatoeba_data.py


训练集和验证集分成了两个文件: text.txt 以及 labels.txt。text.txt 文件的每行包含了文本序列，词之间用空格分割:
[WORD] [SPACE] [WORD] [SPACE] [WORD], 例如:

  ::
    
    when is the next flight to new york
    the next flight is ...
    ...

文件 labels.txt 包含了 text.txt 中每个词的标签(label), 标注用空格分割.
在 labels.txt 文件中的每个标签包含两个符号:

* 标签的第一个符号表示这个词后面应该跟什么标点符号 (其中 ``O`` 表示不需要标点符号);
* 第二个符号决定了这个词是否要大写(其中 ``U`` 说明这个词需要大写， ``O`` 表示不需要大写)

我们在这个任务中只考虑逗号，句号和问号。剩下的标点符号都去除了。
labels.txt 文件的每行都应该是下面这个格式的: 
[LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (labels.txt). 比如，在上面的 text.txt 文件中的标签应该是:

::
    
    OU OO OO OO OO OO OU ?U 
    OU OO OO OO ...
    ...

这个任务所有可能的标签是: ``OO``, ``,O``, ``.O``, ``?O``, ``OU``, ``,U``, ``.U``, ``?U``.

代码概览
-------------

首先, 设置一些必须的参数:

    .. code-block:: python
        
        DATA_DIR = "PATH_TO_WHERE_THE_DATA_IS"
        WORK_DIR = "PATH_TO_WHERE_TO_STORE_CHECKPOINTS_AND_LOGS"
        PRETRAINED_BERT_MODEL = "bert-base-uncased"

        # 模型参数
        BATCHES_PER_STEP = 1
        BATCH_SIZE = 128
        CLASSIFICATION_DROPOUT = 0.1
        MAX_SEQ_LENGTH = 64
        NUM_EPOCHS = 10
        LEARNING_RATE = 0.00002
        LR_WARMUP_PROPORTION = 0.1
        OPTIMIZER = "adam"
        STEP_FREQ = 200 # 决定了 loss 多久打印一次，checkpoint 多久保存一次
        PUNCT_NUM_FC_LAYERS = 3
        NUM_SAMPLES = 100000

下载，预处理一部分的数据集 (Tatoeba collection of sentences), 运行:

.. code-block:: bash
        
        python get_tatoeba_data.py --data_dir DATA_DIR --num_sample NUM_SAMPLES

接着，我们需要用支持的后端创建 neural factory。 这个教程假设你在单卡 GPU 上训练，混精度 (``optimization_level="O1"``)。如果你不想用混精度训练，设置 ``optimization_level`` 为 ``O0``。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=None,
                                           optimization_level="O1",
                                           log_dir=WORK_DIR,
                                           placement=nemo.core.DeviceType.GPU)

然后，定义我们的分词器和 BERT 模型。如果你用标准的 BERT，你可以这么做。想要看所有 BERT O型的名字，可以查看 ``nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = nemo.collections.nlp.data.NemoBertTokenizer(pretrained_model=PRETRAINED_BERT_MODEL)
        bert_model = nemo_nlp.nm.trainables.huggingface.BERT(
            pretrained_model_name=PRETRAINED_BERT_MODEL)

现在, 创建验证和训练的数据层:

    .. code-block:: python

        train_data_layer = nemo_nlp.nm.data_layers.PunctuationCapitalizationDataLayer(
                                            tokenizer=tokenizer,
                                            text_file=os.path.join(DATA_DIR, 'text_train.txt'),
                                            label_file=os.path.join(DATA_DIR, 'labels_train.txt'),
                                            max_seq_length=MAX_SEQ_LENGTH,
                                            batch_size=BATCH_SIZE)

        punct_label_ids = train_data_layer.dataset.punct_label_ids
        capit_label_ids = train_data_layer.dataset.capit_label_ids

        hidden_size = bert_model.hidden_size

        # 注意你需要指定 punct_label_ids 和 capit_label_ids  - 它们是在创建train_data_layer
        # 映射标签到标签id(label_ids)时候生成的
        # 目的是为了确保映射是正确的，
        # 防止一些训练集的标签在验证集上丢失
        eval_data_layer = nemo_nlp.BertPunctuationCapitalizationDataLayer(
                                            tokenizer=tokenizer,
                                            text_file=os.path.join(DATA_DIR, 'text_dev.txt'),
                                            label_file=os.path.join(DATA_DIR, 'labels_dev.txt'),
                                            max_seq_length=MAX_SEQ_LENGTH,
                                            batch_size=BATCH_SIZE,
                                            punct_label_ids=punct_label_ids,
                                            capit_label_ids=capit_label_ids)


现在，在预训练 BERT 模型上创建标签和大写分类器并且定义这个任务的损失函数:

  .. code-block:: python

      punct_classifier = TokenClassifier(
                                         hidden_size=hidden_size,
                                         num_classes=len(punct_label_ids),
                                         dropout=CLASSIFICATION_DROPOUT,
                                         num_layers=PUNCT_NUM_FC_LAYERS,
                                         name='Punctuation')

      capit_classifier = TokenClassifier(hidden_size=hidden_size,
                                         num_classes=len(capit_label_ids),
                                         dropout=CLASSIFICATION_DROPOUT,
                                         name='Capitalization')


      # 如果你不想在标点符号任务上用加权损失函数，设置 class_weights=None
      punct_label_freqs = train_data_layer.dataset.punct_label_frequencies
      class_weights = nemo.collections.nlp.data.datasets.datasets_utils.calc_class_weights(punct_label_freqs)

      # 定义损失函数
      punct_loss = CrossEntropyLossNM(logits_dim=3, weight=class_weights)
      capit_loss = CrossEntropyLossNM(logits_dim=3)
      task_loss = LossAggregatorNM(num_inputs=2)


下面，通过预训练的 BERT 模型，我们传递数据层的输出给到分类器:

  .. code-block:: python

      input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = train_data_layer()

      hidden_states = bert_model(input_ids=input_ids,
                            token_type_ids=input_type_ids,
                            attention_mask=input_mask)

      punct_logits = punct_classifier(hidden_states=hidden_states)
      capit_logits = capit_classifier(hidden_states=hidden_states)

      punct_loss = punct_loss(logits=punct_logits,
                              labels=punct_labels,
                              loss_mask=loss_mask)
      capit_loss = capit_loss(logits=capit_logits,
                              labels=capit_labels,
                              loss_mask=loss_mask)
      task_loss = task_loss(loss_1=punct_loss,
                            loss_2=capit_loss)

      eval_input_ids, eval_input_type_ids, eval_input_mask, _, eval_subtokens_mask, eval_punct_labels, eval_capit_labels\
          = eval_data_layer()

      hidden_states = bert_model(input_ids=eval_input_ids,
                                 token_type_ids=eval_input_type_ids,
                                 attention_mask=eval_input_mask)

      eval_punct_logits = punct_classifier(hidden_states=hidden_states)
      eval_capit_logits = capit_classifier(hidden_states=hidden_states)



现在，我们设置我们的回调函数。我们用3个回调函数:

* `SimpleLossLoggerCallback` 打印训练过程中的损失函数值
* `EvaluatorCallback` 计算验证集上的数据指标
* `CheckpointCallback` 用来保存和还原 checkpoints

    .. code-block:: python

        callback_train = nemo.core.SimpleLossLoggerCallback(
        tensors=[task_loss, punct_loss, capit_loss, punct_logits, capit_logits],
        print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
        step_freq=STEP_FREQ)

        train_data_size = len(train_data_layer)

        # 如果你用多 GPUs，这行应该是
        # train_data_size / (batch_size * batches_per_step * num_gpus)
        steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

        # 回调评估模型
        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_punct_logits,
                          eval_capit_logits,
                          eval_punct_labels,
                          eval_capit_labels,
                          eval_subtokens_mask],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x,
                                                                          punct_label_ids,
                                                                          capit_label_ids),
            eval_step=steps_per_epoch)

        # 回调保存 checkpoints
        ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                                     step_freq=STEP_FREQ)

最后，定义学习率策略和我们的优化器，开始训练:

    .. code-block:: python

        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[task_loss],
                 callbacks=[callback_train, callback_eval, ckpt_callback],
                 lr_policy=lr_policy,
                 batches_per_step=BATCHES_PER_STEP,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

推理
---------

为了看看模型的推理预测，我们在一些样本上运行推理。我们需要定义一个数据层，就像我们为训练和验证评估那样创建的数据层。

.. code-block:: python

    queries = ['can i help you',
               'yes please',
               'we bought four shirts from the nvidia gear store in santa clara',
               'we bought four shirts one mug and ten thousand titan rtx graphics cards',
               'the more you buy the more you save']
    infer_data_layer = nemo_nlp.nm.data_layers.BertTokenClassificationInferDataLayer(
                                                            queries=queries,
                                                            tokenizer=tokenizer,
                                                            max_seq_length=MAX_SEQ_LENGTH,
                                                            batch_size=1)


运行推理，基于训练结果加上标点符号和单词大写:

.. code-block:: python

    input_ids, input_type_ids, input_mask, _, subtokens_mask = infer_data_layer()

    hidden_states = bert_model(input_ids=input_ids,
                                          token_type_ids=input_type_ids,
                                          attention_mask=input_mask)
    punct_logits = punct_classifier(hidden_states=hidden_states)
    capit_logits = capit_classifier(hidden_states=hidden_states)

    evaluated_tensors = nf.infer(tensors=[punct_logits, capit_logits, subtokens_mask],
                                 checkpoint_dir=WORK_DIR + '/checkpoints')



    # 帮助函数
    def concatenate(lists):
        return np.concatenate([t.cpu() for t in lists])

    punct_ids_to_labels = {punct_label_ids[k]: k for k in punct_label_ids}
    capit_ids_to_labels = {capit_label_ids[k]: k for k in capit_label_ids}

    punct_logits, capit_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]
    punct_preds = np.argmax(punct_logits, axis=2)
    capit_preds = np.argmax(capit_logits, axis=2)

    for i, query in enumerate(queries):
        logging.info(f'Query: {query}')

        punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
        capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
        words = query.strip().split()
        if len(punct_pred) != len(words) or len(capit_pred) != len(words):
            raise ValueError('Pred and words must be of the same length')

        output = ''
        for j, w in enumerate(words):
            punct_label = punct_ids_to_labels[punct_pred[j]]
            capit_label = capit_ids_to_labels[capit_pred[j]]

            if capit_label != 'O':
                w = w.capitalize()
            output += w
            if punct_label != 'O':
                output += punct_label
            output += ' '
        logging.info(f'Combined: {output.strip()}\n')

预测结果:
    
    ::

        Query: can i help you
        Combined: Can I help you?

        Query: yes please
        Combined: Yes, please.

        Query: we bought four shirts from the nvidia gear store in santa clara
        Combined: We bought four shirts from the Nvidia gear store in Santa Clara.

        Query: we bought four shirts one mug and ten thousand titan rtx graphics cards
        Combined: We bought four shirts, one mug, and ten thousand Titan Rtx graphics cards.

        Query: the more you buy the more you save
        Combined: The more you buy, the more you save.

.. _punct_scripts:

训练和推理脚本
------------------------------

运行提供的训练脚本:

.. code-block:: bash

    python examples/nlp/token_classification/punctuation_capitalization.py --data_dir path_to_data --pretrained_bert_model=bert-base-uncased --work_dir path_to_output_dir

运行推理:

.. code-block:: bash

    python examples/nlp/token_classification/punctuation_capitalization_infer.py --punct_labels_dict path_to_data/punct_label_ids.csv --capit_labels_dict path_to_data/capit_label_ids.csv --work_dir path_to_output_dir/checkpoints/

注意, punct_label_ids.csv 和 capit_label_ids.csv 文件在训练的时候会生成并且存在 data_dir 文件目录下。

多 GPU 训练
------------------

在多张 GPU 上训练，运行

.. code-block:: bash

    export NUM_GPUS=2
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/nlp/token_classification/punctuation_capitalization.py --num_gpus $NUM_GPUS --data_dir path_to_data
