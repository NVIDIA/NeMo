教程
========

在这个教程中，我们会在 SQuAD 数据集上训练一个问答系统。模型结构用的是预训练的类 BERT 的模型
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ :cite:`nlp-qa-devlin2018bert`.
这个教程中所有的代码都基于 ``examples/nlp/question_answering/question_answering_squad.py``。


目前支持三个主要的预训练模型，在这些模型上，用 SQuAD 数据集进行问答任务的微调(fine-tuning):
BERT, ALBERT and RoBERTa. 这些预训练模型的 checkpoints 来自 `transformers <https://huggingface.co/transformers>`__ . 除了这些，用户也可以在自定义的
BERT checkpoint 上做微调(fine-tuning)，通过制定 `--bert_checkpoint` 参数。
预训练的主要模型的类型可以用 `--model_type` 指定，具体的某个模型用参数 `--pretrained_model_name` 指定。
已有的预训练模型参见这个列表:
`这里 <https://huggingface.co/transformers/pretrained_models.html>`__. 

.. tip::

    如果要在 NeMo 中预训练 BERT 以及预训练好的模型 checkpoints 请参阅 `BERT pretraining <https://nvidia.github.io/NeMo/zh/nlp/bert_pretraining.html>`__.



准备工作
-------------

**模型细节**

这个模型是训练在 token 层面的分类器上，预测在上下文中答案的开始和结束位置。
损失函数值是答案开始 `S_loss` 和答案结束 `E_loss` 的交叉熵损失函数值:

        `S_loss` + `E_loss`

推理(inference)的时候，使得损失值最小的最长答案范围被用来作为预测的答案。

**数据集** 

模型可以处理下面这种格式的任意数据集:

    * 训练文件: 一个 `json` 文件，结构如下

    {"data":[{"title": "string", "paragraphs": [{"context": "string", "qas": [{"question": "string", "is_impossible": "bool", "id": "number", "answers": [{"answer_start": "number", "text": "string", }]}]}]}]}
    "answers" 可以为空，如果模型要学习的问题的是无解的(impossible)，如果是这样的，需要传入参数 `--version_2_with_negative`

    * 验证集文件: 一个 `json` 文件和训练文件结构一样，
      除了它可以对同一个问题提供多个 "answer" 答案。
     

    * 测试文件: 一个 `json` 文件和训练文件结构一样，
      但它并不要求有 "answers" 这个键值。 

目前我们为其提供预处理脚本的数据集是 SQuAD v1.1 和 v2.0 
可以从这里下载:
数据集 `https://rajpurkar.github.io/SQuAD-explorer/ <https://rajpurkar.github.io/SQuAD-explorer/>`_.
预处理脚本位于 ``examples/nlp/question_answering/get_squad.py``。


代码结构
--------------

首先，初始化神经模块工厂( Neural Module Factory)，它定义了 1) 后端 (PyTorch), 2) 混精度优化等级,
3) GPU的本地秩(local rank), 以及 4) 实验管理器(experiment manager)会创建带时间戳的文件夹来存储 checkpoints，相关输出，日志文件，以及 TensorBoard 的图。

    .. code-block:: python
    
        import nemo
        import nemo.collections.nlp as nemo_nlp
        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                               local_rank=args.local_rank,
                                               optimization_level=args.amp_opt_level,
                                               log_dir=work_dir,
                                               create_tb_writer=True,
                                               files_to_copy=[__file__],
                                               add_time_to_log_dir=True)

接着，我们定义参与到我们的问答分类问题管道中的神经模块:

    * 处理数据: `BertQuestionAnsweringDataLayer` 对裸数据处理成 `SquadDataset` 接受的数据格式。
    
    训练和验证评估(evaluation)都需要它们自己的 `BertQuestionAnsweringDataLayer` 数据层。
    数据层(DataLayer)是一个用来为你的数据集做额外的语义检查并把它转换成数据层神经模块(DataLayerNM)的层。 

    .. code-block:: python

        data_layer = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
                                mode="train",
                                data_file=args.train_file,
                                tokenizer=tokenizer,
                                batch_size=args.batch_size,
                                version_2_with_negative=args.version_2_with_negative,
                                max_query_length=args.max_query_length,
                                max_seq_length=args.max_seq_length,
                                doc_stride=args.doc_stride,
                                use_cache=args.use_data_cache)

        
        data_layer_eval = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
                                mode='eval',
                                data_file=args.eval_file,
                                tokenizer=tokenizer,
                                batch_size=args.batch_size,
                                version_2_with_negative=args.version_2_with_negative,
                                max_query_length=args.max_query_length,
                                max_seq_length=args.max_seq_length,
                                doc_stride=args.doc_stride,
                                use_cache=args.use_data_cache)

    * 加载预训练模型，得到相应输入的隐状态(hidden states)。

    .. code-block:: python
        
        args.pretrained_model_name = "bert-base-uncased"
        model = nemo_nlp.nm.trainables.huggingface.BERT(args.pretrained_model_name)
        # 如果模型是 RoBERTa
        args.pretrained_model_name = "roberta-base"
        model = nemo_nlp.nm.trainables.huggingface.Roberta(args.pretrained_model_name)
        # 或者是 Albert
        args.pretrained_model_name = "albert-base-v1"
        model = nemo_nlp.nm.trainables.huggingface.Albert(args.pretrained_model_name)

    * 定义分词器，这里用  `NemoBertTokenizer` 把文本转换成 BERT 的 tokens。这会按照原始的 BERT 模型那样切分文本。

    .. code-block:: python

        hidden_size = model.hidden_size
        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)


    * 为我们的任务创建分类器的头部(head)。

    .. code-block:: python

        qa_head = nemo_nlp.nm.trainables.TokenClassifier(
                                hidden_size=hidden_size,
                                num_classes=2,
                                num_layers=1,
                                log_softmax=False)

    * 创建损失函数

    .. code-block:: python

        loss_fn = nemo_nlp.nm.losses.SpanningLoss()

    * 为训练和验证评估过程创建管道 

    .. code-block:: python

        # training graph
        input_data = data_layer()
        hidden_states = model(input_ids=input_data.input_ids,
                        token_type_ids=input_data.input_type_ids,
                        attention_mask=input_data.input_mask)

        qa_logits = qa_head(hidden_states=hidden_states)
        loss_outputs = squad_loss(
            logits=qa_logits,
            start_positions=input_data.start_positions,
            end_positions=input_data.end_positions)
        train_tensors = [loss_outputs.loss]

        # 评估图
        input_data_eval = data_layer_eval()

        hidden_states_eval = model(
            input_ids=input_data_eval.input_ids,
            token_type_ids=input_data_eval.input_type_ids,
            attention_mask=input_data_eval.input_mask)

        qa_logits_eval = qa_head(hidden_states=hidden_states_eval)
        loss_outputs_eval = squad_loss(
            logits=qa_logits_eval,
            start_positions=input_data_eval.start_positions,
            end_positions=input_data_eval.end_positions)
        eval_tensors = [input_data_eval.unique_ids, loss_outputs_eval.start_logits, loss_outputs_eval.end_logits]



    * 创建回调，保存 checkpoints，打印训练过程和验证评估结果。

    .. code-block:: python

        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=train_tensors,
            print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=args.step_freq,
            tb_writer=neural_factory.tb_writer)


        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x:
                eval_epochs_done_callback(
                    x, eval_data_layer=data_layer_eval,
                    do_lower_case=args.do_lower_case,
                    n_best_size=args.n_best_size,
                    max_answer_length=args.max_answer_length,
                    version_2_with_negative=args.version_2_with_negative,
                    null_score_diff_threshold=args.null_score_diff_threshold),
                tb_writer=neural_factory.tb_writer,
                eval_step=args.eval_step_freq)

        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=args.save_epoch_freq,
            step_freq=args.save_step_freq)

    * 最后，定义优化器参数，运行整个管道

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)

        nf.train(tensors_to_optimize=train_tensors,
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 lr_policy=lr_policy_fn,
                 optimizer=args.optimizer_kind,
                 optimization_params={"num_epochs": args.num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay})

模型训练
--------------

跑在单张 GPU，运行:
    
    .. code-block:: python

        python question_answering_squad.py \
            ...
            
用多卡跑 SQuAD 问答任务，运行 ``question_answering_squad.py`` ，它位于 ``examples/nlp/question_answering``:

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=8 question_answering_squad.py 
            --train_file <*.json 格式的训练文件>
            --eval_file <*.json 格式的验证评估文件>
            --num_gpus 8
            --work_dir <你想在哪里记录你的实验> 
            --amp_opt_level <amp 优化等级> 
            --pretrained_model_name <模型类型> 
            --bert_checkpoint <预训练的 bert checkpoint>
            --mode "train_eval"
            ...

运行评估:

    .. code-block:: python

        python question_answering_squad.py 
            --eval_file <*.json 格式的验证评估文件>
            --checkpoint_dir <已经训练好的 SQuAD 模型的 checkpoint 的文件夹>
            --mode "eval"
            --output_prediction_file <预测结果的输出文件>
            ...

运行推理:

    .. code-block:: python

        python question_answering_squad.py 
            --test_file <*.json 格式的验证评估文件>
            --checkpoint_dir <已经训练好的 SQuAD 模型的 checkpoint 的文件夹>
            --mode "test"
            --output_prediction_file <预测结果的输出文件>
            ...


参考
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-QA
    :keyprefix: nlp-qa-