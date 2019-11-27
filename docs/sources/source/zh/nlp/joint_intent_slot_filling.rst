教程
========

在这个教程中，我们将使用BERT模型，来实现一个意图识别（intent）和槽填充（slot filling）混合系统，参考自 `BERT for Joint Intent Classification and Slot Filling <https://arxiv.org/abs/1902.10909>`_ :cite:`chen2019bert`。本教程中所有的代码全部来自 ``examples/nlp/joint_intent_slot_with_bert.py``.

我们可以使用 `--pretrained_bert_model` 这个参数，来选择四个预训练好的BERT模型。当前，我们使用的加载预训练模型的脚本均来自 `pytorch_transformers` 。更多预训练好的模型在 `这里 <https://huggingface.co/pytorch-transformers/pretrained_models.html>`__ 。 


写在开头
-------------

**模型细节**
这个模型会一起训练一个句子层面的分类器，和一个符号串层面的槽分类器，通过最小化如下的混合损失:

        intent_loss * intent_loss_weight + slot_loss * (1 - intent_loss_weight)

当 `intent_loss_weight = 0.5` 时, 它等价于最大化:

        p(y | x)P(s1, s2, ..., sn | x)

这里x是一个有n个符号串的序列(x1, x2, ..., xn)，y是x预测出的意图，s1, s2, ..., sn 是对应于x1, x2, ..., xn预测出的槽。

**数据集** 

这个模型可以应用到任意一个符合如下格式的数据集:
    * 输入文件: 一个 `tsv` 文件，第一行为 [sentence][tab][label] 

    * 槽文件: 句子中所有符号串的槽标注，使用空格分隔。槽标注的数量需要与句子中所有符号串的数量保持一致。

当前，我们提供多个数据集合的预处理脚本，包括: ATIS可以通过 `Kaggle <https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk>`_ 进行下载；SNIP对话语言理解数据集，可以通过 `这里 <https://github.com/snipsco/spoken-language-understanding-research-datasets>`__ 获取。预处理脚本在 ``collections/nemo_nlp/nemo_nlp/text_data_utils.py``。


代码结构
--------------

首先，我们初始化Neural Module Factory，需要定义1、后端（Pytorch或是TensorFlow)；2、混合精度优化的级别；3、本地GPU的序列号；4、一个实验的管理器，用于创建文件夹来保存相应的checkpoint、输出、日志文件和TensorBoard的图。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(
                        backend=nemo.core.Backend.PyTorch,
                        local_rank=args.local_rank,
                        optimization_level=args.amp_opt_level,
                        log_dir=work_dir,
                        create_tb_writer=True,
                        files_to_copy=[__file__])

我们定义分词器，它可以将文本转换成符号串，这里使用来自 `pytorch_transformers`的内置分词器。其将使用BERT模型的映射，把文本转成相应的符号串。

    .. code-block:: python

        from pytorch_transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

接着，我们定义所有的神经网络模块，加入到意图识别和槽填充混合系统的流程中。
    
    * 处理数据: `nemo_nlp/nemo_nlp/text_data_utils.py` 中的 `JointIntentSlotDataDesc` 类，用于将源数据处理成 `BertJointIntentSlotDataset` 支持的类型。当前，它支持SNIPS和ATIS两种格式的数据，当你也可以实现预处理脚本，来支持任意格式的数据。 

    JointIntentSlotDataDesc 对象包含例如 `self.train_file`, `self.train_slot_file`, `self.eval_file`, `self.eval_slot_file`,  `self.intent_dict_file` 和 `self.slot_dict_file`等信息。

    .. code-block:: python

        data_desc = JointIntentSlotDataDesc(
            args.dataset_name, args.data_dir, args.do_lower_case)


    * 数据集: 将数据转换成DataLayerNM可以接收的格式.

    .. code-block:: python

        def get_dataset(data_desc, mode, num_samples):
            nf.logger.info(f"Loading {mode} data...")
            data_file = getattr(data_desc, mode + '_file')
            slot_file = getattr(data_desc, mode + '_slot_file')
            shuffle = args.shuffle_data if mode == 'train' else False
            return nemo_nlp.BertJointIntentSlotDataset(
                input_file=data_file,
                slot_file=slot_file,
                pad_label=data_desc.pad_label,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                num_samples=num_samples,
                shuffle=shuffle)


        train_dataset = get_dataset(data_desc, 'train', args.num_train_samples)
        eval_dataset = get_dataset(data_desc, 'eval', args.num_eval_samples)

    * DataLayer: 一个单独的层，可以用于在你的数据集中进行语义检查，并将它转换到DataLayerNM中。你需要定义 `input_ports` 和 `output_ports` 。

    .. code-block:: python

        data_layer = nemo_nlp.BertJointIntentSlotDataLayer(dataset,
                                                batch_size=batch_size,
                                                num_workers=0,
                                                local_rank=local_rank)

        ids, type_ids, input_mask, slot_mask, intents, slots = data_layer()


    * 加载预训练好的模型，并得到相应输入的隐层状态。

    .. code-block:: python

        hidden_states = pretrained_bert_model(input_ids=ids,
                                              token_type_ids=type_ids,
                                              attention_mask=input_mask)


    * 为我们的任务创建一个分类器。

    .. code-block:: python

        classifier = nemo_nlp.JointIntentSlotClassifier(
                                        hidden_size=hidden_size,
                                        num_intents=num_intents,
                                        num_slots=num_slots,
                                        dropout=args.fc_dropout)

        intent_logits, slot_logits = classifier(hidden_states=hidden_states)


    * 创建损失函数。 

    .. code-block:: python

        loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=num_slots)

        loss = loss_fn(intent_logits=intent_logits,
                       slot_logits=slot_logits,
                       input_mask=input_mask,
                       intents=intents,
                       slots=slots)


    * 创建相应的callbacks，来保存checkpoints，打印训练过程和测试结果。

    .. code-block:: python

        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: str(np.round(x[0].item(), 3)),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=steps_per_epoch)

        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(
                x, y, data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=args.save_epoch_freq,
            step_freq=args.save_step_freq)

    * 最后，我们定义优化器的参数，并开始训练流程。

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)
        nf.train(tensors_to_optimize=[train_loss],
             callbacks=[callback_train, callback_eval, ckpt_callback],
             lr_policy=lr_policy_fn,
             optimizer=args.optimizer_kind,
             optimization_params={"num_epochs": num_epochs,
                                  "lr": args.lr,
                                  "weight_decay": args.weight_decay})

模型训练
--------------

为了训练一个意图识别和槽填充的混合任务，运行 ``nemo/examples/nlp`` 下的脚本 ``joint_intent_slot_with_bert.py`` ：

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=2 joint_intent_slot_with_bert.py \
            --data_dir <path to data>
            --work_dir <where you want to log your experiment> \
            --max_seq_length \
            --optimizer_kind 
            ...

测试的话，需要运行：

    .. code-block:: python

        python -m joint_intent_slot_infer.py \
            --data_dir <path to data> \
            --work_dir <path to checkpoint folder>

对一个检索进行测试，需要运行：
    
    .. code-block:: python

        python -m joint_intent_slot_infer.py \
            --work_dir <path to checkpoint folder>
            --query <query>


参考文献
----------

.. bibliography:: joint_intent_slot.bib
    :style: plain
