Tutorial
========

In this tutorial, we are going to implement a joint intent and slot filling system with pretrained BERT model based on
`BERT for Joint Intent Classification and Slot Filling <https://arxiv.org/abs/1902.10909>`_ :cite:`nlp-slot-chen2019bert`.
All code used in this tutorial is based on ``examples/nlp/intent_detection_slot_tagging/joint_intent_slot_with_bert.py``.

There are a variety pre-trained BERT models that we can select from using the argument `--pretrained_bert_model`. We're currently
using the script for loading pre-trained models from `pytorch_transformers`. See the list of available pre-trained models
`here <https://huggingface.co/pytorch-transformers/pretrained_models.html>`__.

.. tip::

    For pretraining BERT in NeMo and pretrained model checkpoints go to `BERT pretraining <https://nvidia.github.io/NeMo/nlp/bert_pretraining.html>`__.


Preliminaries
-------------

**Model details**
This model jointly train the sentence-level classifier for intents and token-level classifier for slots by minimizing the combined loss of the two classifiers:

        intent_loss * intent_loss_weight + slot_loss * (1 - intent_loss_weight)

When `intent_loss_weight = 0.5`, this loss jointly maximizes:

        p(y | x)P(s1, s2, ..., sn | x)

with x being the sequence of n tokens (x1, x2, ..., xn), y being the predicted intent for x, and s1, s2, ..., sn being the predicted slots corresponding to x1, x2, ..., xn.

**Datasets.**

This model can work with any dataset that follows the format:
    * input file: a `tsv` file with the first line as a header [sentence][tab][label]

    * slot file: slot labels for all tokens in the sentence, separated by space. The length of the slot labels should be the same as the length of all tokens in sentence in input file.

Currently, the datasets that we provide pre-processing script for include ATIS which can be downloaded
from `Kaggle <https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk>`_ and the SNIPS spoken language understanding research dataset which can be
requested from `here <https://github.com/snipsco/spoken-language-understanding-research-datasets>`__.
You can find the pre-processing script in ``collections/nemo_nlp/nemo_nlp/data/datasets/utils.py``.


Code structure
--------------

First, we instantiate Neural Module Factory which defines 1) backend (PyTorch or TensorFlow), 2) mixed precision optimization level,
3) local rank of the GPU, and 4) an experiment manager that creates a timestamped folder to store checkpoints, relevant outputs, log files, and TensorBoard graphs.

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

We define the tokenizer which transforms text into BERT tokens, using a built-in tokenizer by `pytorch_transformers`.
This will tokenize text following the mapping of the original BERT model.

    .. code-block:: python

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

Next, we define all Neural Modules participating in our joint intent slot filling classification pipeline.

    * Process data: the `JointIntentSlotDataDesc` class in `nemo/collections/nlp/data/datasets/joint_intent_slot_dataset/data_descriptor.py` is supposed to do the preprocessing of raw data into the format data supported by `BertJointIntentSlotDataset`. Currently, it supports SNIPS and ATIS raw datasets, but you can also write your own preprocessing scripts for any dataset.

    .. code-block:: python

        from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset import JointIntentSlotDataDesc
        data_desc = JointIntentSlotDataDesc(
            args.data_dir, args.do_lower_case, args.dataset_name, args.none_slot_label, args.pad_label
        )


    * Load the pretrained BERT model to encode the corresponding inputs.

    .. code-block:: python

        from nemo.collections.nlp.nm.trainables.common.huggingface import BERT
        pretrained_bert_model = BERT(pretrained_model_name=args.pretrained_bert_model)

    * Create the classifier heads for our task.

    .. code-block:: python

        from nemo.collections.nlp.nm.trainables import JointIntentSlotClassifier
        classifier = JointIntentSlotClassifier(
            hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots, dropout=args.fc_dropout
        )

    * Create loss functions for intent detection and slot filling then and use loss aggregator module to merge them

    .. code-block:: python

        from nemo.backends.pytorch.common.losses import CrossEntropyLossNM, LossAggregatorNM
        intent_loss_fn = CrossEntropyLossNM(logits_dim=2)
        slot_loss_fn = CrossEntropyLossNM(logits_dim=3)
        total_loss_fn = LossAggregatorNM(num_inputs=2, weights=[args.intent_loss_weight, 1.0 - args.intent_loss_weight])

    * Create the pipelines for the train and evaluation processes. Each pipeline creates its own data layer (BertJointIntentSlotDataLayer). DataLayer is an extra layer to do the semantic checking for your dataset and convert it into DataLayerNM. You have to define `input_ports` and `output_ports`.

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

    * Create relevant callbacks for saving checkpoints, printing training progresses and evaluating results.

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

    * Finally, we define the optimization parameters and run the whole pipeline.

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

Model training
--------------

To train a joint intent slot filling model, run ``joint_intent_slot_with_bert.py`` located at ``examples/nlp/intent_detection_slot_tagging/joint_intent_slot_with_bert.py``:

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=2 joint_intent_slot_with_bert.py \
            --data_dir <path to data>
            --work_dir <where you want to log your experiment> \

To do inference, run:

    .. code-block:: python

        python joint_intent_slot_infer.py \
            --data_dir <path to data> \
            --work_dir <path to checkpoint folder>

To do inference on a single query, run:

    .. code-block:: python

        python joint_intent_slot_infer.py \
            --work_dir <path to checkpoint folder>
            --query <query>


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-SLOT
    :keyprefix: nlp-slot-
