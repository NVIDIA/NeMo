教程
====

在这个教程中，我们将使用 BERT 模型，来实现一个意图识别 (intent classification) 和槽填充 (slot filling) 混合系统，参考自 `BERT for Joint Intent Classification and Slot Filling <https://arxiv.org/abs/1902.10909>`_ :cite:`nlp-slot-chen2019bert` 。本教程中所有的代码全部来自 ``examples/nlp/intent_detection_slot_tagging/joint_intent_slot_with_bert.py``。

我们可以使用 `--pretrained_bert_model` 这个参数，来选择多个预训练好的 BERT 模型。当前，我们使用的加载预训练模型的脚本均来自 `pytorch_transformers` 。更多预训练好的模型在 `这里下载 <https://huggingface.co/pytorch-transformers/pretrained_models.html>`_ 。

.. tip::

    在 NeMo 中进行BERT的预训练以及预训练好的模型checkpoints可以参见 `BERT pretraining <https://nvidia.github.io/NeMo/nlp/bert_pretraining.html>`__ 。


写在开头
--------

**模型细节**
这个模型会一起训练一个句子层级的分类器，和一个符号串层面的槽 (slot) 分类器，通过最小化如下的混合损失：

        intent_loss * intent_loss_weight + slot_loss * (1 - intent_loss_weight)

当 `intent_loss_weight = 0.5` 时，它等价于最大化:

        p(y | x)P(s1, s2, ..., sn | x)

这里 x 是一个有 n 个符号串的序列 (x1, x2, ..., xn)， y 是 x 预测出的意图，s1, s2, ..., sn 是对应于 x1, x2, ..., xn 预测出的槽。

**数据集**

这个模型可以应用到任意一个符合如下格式的数据集:

    * 输入文件: 一个 `tsv` 文件，第一行为 [sentence][tab][label]
    * 槽文件: 句子中所有符号串的槽标注，使用空格分隔。槽标注的数量需要与句子中所有符号串的数量保持一致。

当前，我们提供多个数据集合的预处理脚本，包括: ATIS，可以通过 `Kaggle <https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk>`_ 进行下载；SNIP对话语言理解数据集，可以通过 `这里 <https://github.com/snipsco/spoken-language-understanding-research-datasets>`_ 获取。通过把数据集名称这个参数设置成['atis', 'snips-light', 'snips-speak', 'snips-all']，你就可以将其转换成 NeMo 中的格式。


代码结构
--------

首先，我们初始化 Neural Module Factory，需要定义，1、后端 (PyTorch 或者 Tensorflow)；2、混合精度优化的级别；3、本地 GPU 的序列号；4、一个实验的管理器，用于创建文件夹来保存相应的 checkpoint、输出、日志文件和 TensorBoard 的图。

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=args.local_rank,
            optimization_level=args.amp_opt_level,
            log_dir=work_dir,
            create_tb_writer=True,
            files_to_copy=[__file__],
            add_time_to_log_dir=True,
        )

我们定义分词器，它可以将文本转换成符号串，这里使用来自 `transformers` 的内置分词器。其将使用 BERT 模型的映射，把文本转成相应的符号串。

    .. code-block:: python

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

接着，我们定义所有的神经网络模块，加入到意图识别和槽填充混合系统的流程中。

    * 处理数据: `nemo/collections/nlp/data/datasets/joint_intent_slot_dataset/data_descriptor.py` 中的 `JointIntentSlotDataDesc` 类，用于将源数据处理成 `BertJointIntentSlotDataset` 支持的类型。当前，它支持 SNIPS 和 ATIS 两种格式的数据，当你也可以实现预处理脚本，来支持任意格式的数据。

    .. code-block:: python

        from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset import JointIntentSlotDataDesc
        data_desc = JointIntentSlotDataDesc(
            args.data_dir, args.do_lower_case, args.dataset_name, args.none_slot_label, args.pad_label
        )

    * 加载预训练好的 BERT 模型来对相应的输入进行编码。

    .. code-block:: python

        from nemo.collections.nlp.nm.trainables.common.huggingface import BERT
        pretrained_bert_model = BERT(pretrained_model_name=args.pretrained_bert_model)

    * 为我们的任务创建分类器。

    .. code-block:: python

        from nemo.collections.nlp.nm.trainables import JointIntentSlotClassifier
        classifier = JointIntentSlotClassifier(
            hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots, dropout=args.fc_dropout
        )

    * 为意图检测和槽填充创建损失函数，并使用损失累积模块将二者合并。

    .. code-block:: python

        from nemo.backends.pytorch.common.losses import CrossEntropyLossNM, LossAggregatorNM
        intent_loss_fn = CrossEntropyLossNM(logits_dim=2)
        slot_loss_fn = CrossEntropyLossNM(logits_dim=3)
        total_loss_fn = LossAggregatorNM(num_inputs=2, weights=[args.intent_loss_weight, 1.0 - args.intent_loss_weight])

    * 创建训练和测试过程的管道。每个管道拥有自己的数据层 (BertJointIntentSlotDataLayer)。数据层是一个单独用于数据语义检测的层，并可以把数据转换到 DataLayerNM 中，你需要定义 `input_ports` 和 `output_ports`。

    .. code-block:: python

        from nemo.collections.nlp.nm.data_layers import BertJointIntentSlotDataLayer
        def create_pipeline(num_samples=-1, batch_size=32, data_prefix='train', is_training=True, num_gpus=1):
            logging.info(f"Loading {data_prefix} data...")
            data_file = f'{data_desc.data_dir}/{data_prefix}.tsv'
            slot_file = f'{data_desc.data_dir}/{data_prefix}_slots.tsv'
            shuffle = args.shuffle_data if is_training else False

            data_layer = BertJointIntentSlotDataLayer(
                input_file=data_file,
                slot_file=slot_file,
                pad_label=data_desc.pad_label,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                num_samples=num_samples,
                shuffle=shuffle,
                batch_size=batch_size,
                ignore_extra_tokens=args.ignore_extra_tokens,
                ignore_start_end=args.ignore_start_end,
            )

            input_data = data_layer()
            data_size = len(data_layer)

            logging.info(f'The length of data layer is {data_size}')

            if data_size < batch_size:
                logging.warning("Batch_size is larger than the dataset size")
                logging.warning("Reducing batch_size to dataset size")
                batch_size = data_size

            steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
            logging.info(f"Steps_per_epoch = {steps_per_epoch}")

            hidden_states = pretrained_bert_model(
                input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
            )

            intent_logits, slot_logits = classifier(hidden_states=hidden_states)

            intent_loss = intent_loss_fn(logits=intent_logits, labels=input_data.intents)
            slot_loss = slot_loss_fn(logits=slot_logits, labels=input_data.slots, loss_mask=input_data.loss_mask)
            total_loss = total_loss_fn(loss_1=intent_loss, loss_2=slot_loss)

            if is_training:
                tensors_to_evaluate = [total_loss, intent_logits, slot_logits]
            else:
                tensors_to_evaluate = [
                    intent_logits,
                    slot_logits,
                    input_data.intents,
                    input_data.slots,
                    input_data.subtokens_mask,
                ]

            return tensors_to_evaluate, total_loss, steps_per_epoch, data_layer


        train_tensors, train_loss, train_steps_per_epoch, _ = create_pipeline(
            num_samples=args.num_train_samples,
            batch_size=args.batch_size,
            data_prefix=args.train_file_prefix,
            is_training=True,
            num_gpus=args.num_gpus,
        )
        eval_tensors, _, _, eval_data_layer = create_pipeline(
            num_samples=args.num_eval_samples,
            batch_size=args.batch_size,
            data_prefix=args.eval_file_prefix,
            is_training=False,
            num_gpus=args.num_gpus,
        )

    * 创建相应的 callbacks ，来保存 checkpoints，打印训练过程和测试结果。

    .. code-block:: python

        from nemo.collections.nlp.callbacks.joint_intent_slot_callback import eval_epochs_done_callback, eval_iter_callback
        from nemo.core import CheckpointCallback, SimpleLossLoggerCallback
        train_callback = SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: logging.info(str(round(x[0].item(), 3))),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=steps_per_epoch,
        )

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y, data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch,
        )

        ckpt_callback = CheckpointCallback(
            folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
        )

    * 最后，我们定义优化器的参数，并开始训练流程。

    .. code-block:: python

        from nemo.utils.lr_policies import get_lr_policy
        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
        )

        nf.train(
            tensors_to_optimize=[train_loss],
            callbacks=[train_callback, eval_callback, ckpt_callback],
            lr_policy=lr_policy_fn,
            optimizer=args.optimizer_kind,
            optimization_params={"num_epochs": args.num_epochs, "lr": args.lr, "weight_decay": args.weight_decay},
        )


模型训练
--------

为了训练一个意图识别和槽填充的混合任务，运行 ``examples/nlp/intent_detection_slot_tagging/joint_intent_slot_with_bert.py`` 下的脚本 ``joint_intent_slot_with_bert.py`` ：

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=2 joint_intent_slot_with_bert.py \
            --data_dir <path to data>
            --work_dir <where you want to log your experiment> \

测试的话，需要运行：

    .. code-block:: python

        python joint_intent_slot_infer.py \
            --data_dir <path to data> \
            --work_dir <path to checkpoint folder>

对一个检索进行测试，需要运行：

    .. code-block:: python

        python joint_intent_slot_infer.py \
            --work_dir <path to checkpoint folder>
            --query <query>


参考文献
--------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-SLOT
    :keyprefix: nlp-slot-
