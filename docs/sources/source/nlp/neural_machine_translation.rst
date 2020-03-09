Tutorial
========

In this tutorial we are going to implement Neural Machine Translation (NMT) system based on
`Transformer encoder-decoder architecture <https://arxiv.org/abs/1706.03762>`_ :cite:`nlp-nmt-vaswani2017attention`.
All code used in this tutorial is based on ``examples/nlp/neural_machine_translation/machine_translation_tutorial.py``.

Preliminaries
-------------

**Dataset.** We use WMT16 English-German dataset which consists of approximately 4.5 million sentence pairs before preprocessing.
To clean the dataset we remove all sentence pairs such that:

    * The length of either source or target is greater than 128 or smaller than 3 tokens.
    * Absolute difference between source and target is greater than 25 tokens.
    * One sentence is more than 2.5 times longer than the other.
    * Target sentence is the exact copy of the source sentence :cite:`nlp-nmt-ott2018analyzing`.

We use newstest2013 for development and newstest2014 for testing. All datasets, as well as the tokenizer model can be downloaded from
`here <https://drive.google.com/open?id=1AErD1hEg16Yt28a-IGflZnwGTg9O27DT>`__. In the following steps, we assume that all data is located at **<path_to_data>**.

**Resources.** Training script ``examples/nlp/neural_machine_translation/machine_translation_tutorial.py`` used in this tutorial allows to train Transformer-big architecture
to **29.2** BLEU / **28.5** SacreBLEU on newstest2014 in approximately 15 hours on NVIDIA's DGX-1 with 16GB Volta GPUs.
This setup can also be replicated with fewer resources by using more steps of gradient accumulation :cite:`nlp-nmt-ott2018scaling`.

.. tip::
    Launching training script without any arguments will run training on much smaller dataset (newstest2013) of 3000 sentence pairs and validate on the subset
    of this dataset consisting of 100 sentence pairs. This is useful for debugging purposes: if everything is set up correctly, validation BLEU will reach >99
    and training / validation losses will go to <1.5 pretty fast.

Code overview
-------------

First of all, we instantiate Neural Module Factory which defines 1) backend, 2) mixed precision optimization level, and 3) local rank of the GPU.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=args.local_rank,
                                           optimization_level=args.amp_opt_level,
                                           log_dir=args.work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__])

We define tokenizer which allows to transform input text into tokens. In this tutorial, we use joint
`Byte Pair Encodings (BPE) <https://arxiv.org/abs/1508.07909>`_ :cite:`nlp-nmt-sennrich2015neural` trained on WMT16 En-De corpus with
`YouTokenToMe library <https://github.com/VKCOM/YouTokenToMe>`_. In contrast to the models presented in the literature (which usually have vocabularies of size 30000+),
we work with 4x smaller vocabulary of 8192 BPEs. It achieves the same level of performance but allows to increase the batch size by 20% which in turn leads to faster convergence.


    .. code-block:: python

        tokenizer = nemo_nlp.data.YouTokenToMeTokenizer(
            model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)


    .. tip::
        To leverage the best GPU utilization and mixed precision speedup, make sure that the vocabulary size (as well as all sizes in the model) is divisible by 8.

If the source language differs from the target language a lot, then we should use different tokenizers for them. For example, if the source language is English and the target language is Chinese, we can use YouTokenToMeTokenizer for source and CharTokenizer for target. This means the input of the model are English BPEs and the output of the model are Chinese characters.


    .. code-block:: python

        src_tokenizer = nemo_nlp.data.YouTokenToMeTokenizer(
            model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
        tgt_tokenizer = nemo_nlp.data.CharTokenizer(
            vocab_path=f"{args.data_dir}/{args.tgt_tokenizer_model}")

    .. tip::
        You should pass the path of the vocabulary file to the CharTokenizer. The vocabulary file should contain the characters of the corresponding language.

Next, we define all Neural Modules necessary for our model:

    * Transformer Encoder and Decoder.
    * `TokenClassifier` for mapping output of the decoder into probability distribution over vocabulary.
    * Beam Search module for generating translations.
    * Loss function (cross entropy with label smoothing regularization).

    .. code-block:: python

        encoder = nemo_nlp.nm.trainables.TransformerEncoderNM(**encoder_params)
        decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(**decoder_params)
        log_softmax = nemo_nlp.nm.trainables.TokenClassifier(**token_classifier_params)
        beam_search = nemo_nlp.nm.trainables.BeamSearchTranslatorNM(**beam_search_params)
        loss = nemo_nlp.nm.losses.SmoothedCrossEntropyLoss(pad_id=tgt_tokenizer.pad_id, label_smoothing=args.label_smoothing)

Following `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`nlp-nmt-press2016using`, we also tie the parameters of embedding and softmax layers:

    .. code-block:: python

        log_softmax.tie_weights_with(
                encoder,
                weight_names=["mlp.last_linear_layer.weight"],
                name2name_and_transform={
                    "mlp.last_linear_layer.weight": ("embedding_layer.token_embedding.weight", WeightShareTransform.SAME)
                },
            ) 
        decoder.tie_weights_with(
            encoder,
            weight_names=["embedding_layer.token_embedding.weight"],
            name2name_and_transform={
                "embedding_layer.token_embedding.weight": (
                    "embedding_layer.token_embedding.weight",
                    WeightShareTransform.SAME,
                )
            },
        )
        
    .. note::
        You should not tie the parameters if you use different tokenizers for source and target.

Then, we create the pipeline gtom input to output that can be used for both training and evaluation. An important element of this pipeline is the datalayer that
packs input sentences into batches of similar length to minimize the use of padding symbol. Note, that the maximum allowed number of tokens in a batch is given
in **source and target** tokens.

    .. code-block:: python

        def create_pipeline(**args):-
            data_layer = nemo_nlp.nm.data_layers.TranslationDataLayer(**translation_datalayer_params)
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



Next, we define necessary callbacks:

1. `SimpleLossLoggerCallback`: tracking loss during training
2. `EvaluatorCallback`: tracking BLEU score on evaluation dataset at set intervals
3. `CheckpointCallback`: saving model checkpoints

    .. code-block:: python

        from nemo.collections.nlp.callbacks.machine_translation_callbacks import eval_iter_callback, eval_epochs_done_callback

        train_callback = nemo.core.SimpleLossLoggerCallback(...)
        eval_callback = nemo.core.EvaluatorCallback(...)
        ckpt_callback = nemo.core.CheckpointCallback(...)

    .. note::

        The BLEU score is calculated between detokenized translation (generated with beam search) and genuine evaluation dataset. For the sake of completeness,
        we report both  `SacreBLEU <https://github.com/mjpost/sacreBLEU>`_ :cite:`nlp-nmt-post2018call` and
        `tokenized BLEU score <https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl>`_ commonly used in the literature.

Finally, we define the optimization parameters and run the whole pipeline.

    .. code-block:: python

        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.max_steps,
                                     warmup_steps=args.warmup_steps)

        nf.train(tensors_to_optimize=[train_loss],
                 callbacks=[train_callback, eval_callback, ckpt_callback],
                 optimizer=args.optimizer,
                 lr_policy=lr_policy_fn,
                 optimization_params={"num_epochs": max_num_epochs,
                                      "lr": args.lr,
                                      "weight_decay": args.weight_decay,
                                      "betas": (args.beta1, args.beta2)},
                 batches_per_step=args.iter_per_step)


Model training
--------------

To train the Transformer-big model, run ``machine_translation_tutorial.py`` located at ``examples/nlp/neural_machine_translation``:

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=<num_gpus> machine_translation_tutorial.py \
            --data_dir <path_to_data> --src_tokenizer_model bpe8k_yttm.model \
            --eval_datasets valid/newstest2013 --optimizer novograd --lr 0.04 \
            --weight_decay 0.0001 --max_steps 40000 --warmup_steps 4000 \
            --d_model 1024 --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
            --batch_size 12288 --iter_per_step 5


    .. note::

        This command runs training on 8 GPUs with at least 16 GB of memory. If your GPUs have less memory, decrease the **batch_size** parameter.
        To train with bigger batches which do not fit into the memory, increase the **iter_per_step** parameter.

Translation with pretrained model
---------------------------------

1. Put your saved checkpoint (or download good checkpoint which obtains 28.5 SacreBLEU on newstest2014 from
`here <https://ngc.nvidia.com/catalog/models/nvidia:transformer_big_en_de_8k>`__) into **<path_to_ckpt>**.
2. Run ``machine_translation_tutorial.py`` in an interactive mode::

    python machine_translation_tutorial.py --src_tokenizer_model bpe8k_yttm.model \
         --eval_datasets test --optimizer novograd --d_model 1024 \
         --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
         --restore_checkpoint_from <path_to_ckpt> --interactive


   .. image:: interactive_translation.png
       :align: center

References
----------

References
------------------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-NMT
    :keyprefix: nlp-nmt-
