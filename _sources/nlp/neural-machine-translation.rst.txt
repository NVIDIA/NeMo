Tutorial
========

In this tutorial we are going to implement Neural Machine Translation (NMT) system based on `Transformer encoder-decoder architecture <https://arxiv.org/abs/1706.03762>`_ :cite:`vaswani2017attention`. All code used in this tutorial is based on ``examples/nlp/nmt_tutorial.py``.

Preliminaries
-------------

**Dataset.** We use WMT16 English-German dataset which consists of approximately 4.5 million sentence pairs before preprocessing. To clean the dataset we remove all sentence pairs such that:

    * The length of either source or target is greater than 128 or smaller than 3 tokens.
    * Absolute difference between source and target is greater than 25 tokens.
    * One sentence is more than 2.5 times longer than the other.
    * Target sentence is the exact copy of the source sentence :cite:`ott2018analyzing`.

We use newstest2013 for development and newstest2014 for testing. All datasets, as well as the tokenizer model can be downloaded from `here <https://drive.google.com/open?id=1AErD1hEg16Yt28a-IGflZnwGTg9O27DT>`_. In the following steps, we assume that all data is located at **<path_to_data>**.

**Resources.** Training script ``examples/nlp/nmt_tutorial.py`` used in this tutorial allows to train Transformer-big architecture to **29.2** BLEU / **28.5** SacreBLEU on newstest2014 in approximately 15 hours on NVIDIA's DGX-1 with 16GB Volta GPUs. This setup can also be replicated with fewer resources by using more steps of gradient accumulation :cite:`ott2018scaling`.

.. tip::
    Launching training script without any arguments will run training on much smaller dataset (newstest2013) of 3000 sentence pairs and validate on the subset of this dataset consisting of 100 sentence pairs. This is useful for debugging purposes: if everything is set up correctly, validation BLEU will reach >99 and training / validation losses will go to <1.5 pretty fast.

Code overview
-------------

First of all, we instantiate Neural Module Factory which defines 1) backend, 2) mixed precision optimization level, and 3) local rank of the GPU.

    .. code-block:: python

        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=args.local_rank,
            optimization_level=nemo.core.Optimization.mxprO2)

We define tokenizer which allows to transform input text into tokens. In this tutorial, we use joint `Byte Pair Encodings (BPE) <https://arxiv.org/abs/1508.07909>`_ :cite:`sennrich2015neural` trained on WMT16 En-De corpus with `YouTokenToMe library <https://github.com/VKCOM/YouTokenToMe>`_. In contrast to the models presented in the literature (which usually have vocabularies of size 30000+), we work with 4x smaller vocabulary of 8192 BPEs. It achieves the same level of performance but allows to increase the batch size by 20% which in turn leads to faster convergence.


    .. code-block:: python

        tokenizer = nemo_nlp.YouTokenToMeTokenizer(model_path="<path_to_data>/bpe8k_yttm.model")

    .. tip::
        To leverage the best GPU utilization and mixed precision speedup, make sure that the vocabulary size (as well as all sizes in the model) is divisible by 8.

Next, we define all Neural Modules participating in our NMT pipeline:

    * Two data layers (one for training and one for evaluation) which pack input sentences into batches of similar length to minimize the use of padding symbol. Note, that the maximum allowed number of tokens in a batch is given in **source and target** tokens.
    * Transformer Encoder and Decoder.
    * LogSoftmax for mapping output of the decoder into probability distribution over vocabulary.
    * Beam Search module for generating translations.
    * Loss function (cross entropy with label smoothing regularization).

    .. code-block:: python

        train_data_layer = nemo_nlp.TranslationDataLayer(**train_datalayer_params)
        eval_data_layer = nemo_nlp.TranslationDataLayer(**eval_datalayer_params)
        encoder = nemo_nlp.TransformerEncoderNM(**encoder_params)
        decoder = nemo_nlp.TransformerDecoderNM(**decoder_params)
        log_softmax = nemo_nlp.TransformerLogSoftmaxNM(**log_softmax_params)
        beam_search = nemo_nlp.BeamSearchTranslatorNM(**beam_search_params)
        loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(**loss_params)

Following `Press and Wolf, 2016 <https://arxiv.org/abs/1608.05859>`_ :cite:`press2016using`, we also tie the parameters of embedding and softmax layers:

    .. code-block:: python

        log_softmax.log_softmax.dense.weight = encoder.embedding_layer.token_embedding.weight
        decoder.embedding_layer.token_embedding.weight = encoder.embedding_layer.token_embedding.weight

Then, we build the computation graph out of instantiated modules:

    .. code-block:: python

        ########################### Training pipeline ###########################
        src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()
        src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
        tgt_hiddens = decoder(input_ids_tgt=tgt,
                              hidden_states_src=src_hiddens,
                              input_mask_src=src_mask,
                              input_mask_tgt=tgt_mask)
        log_softmax = log_softmax(hidden_states=tgt_hiddens)
        train_loss = loss(log_probs=log_softmax, target_ids=labels)

        ########################## Evaluation pipeline ##########################
        src_, src_mask_, tgt_, tgt_mask_, labels_, sent_ids_ = eval_data_layer()
        src_hiddens_ = encoder(input_ids=src_, input_mask_src=src_mask_)
        tgt_hiddens_ = decoder(input_ids_tgt=tgt_,
                               hidden_states_src=src_hiddens_,
                               input_mask_src=src_mask_,
                               input_mask_tgt=tgt_mask_)
        log_softmax_ = log_softmax(hidden_states=tgt_hiddens_)
        eval_loss = loss(log_probs=log_softmax_, target_ids=labels_)
        beam_trans = beam_search(hidden_states_src=src_hiddens_,
                                 input_mask_src=src_mask_)

Next, we define necessary callbacks for: 1) tracking loss during training, 2) tracking BLEU score on evaluation dataset, 3) saving model checkpoints once in a while.

    .. code-block:: python

        from nemo_nlp.callbacks.translation import eval_iter_callback, eval_epochs_done_callback

        callback_train = nemo.core.SimpleLossLoggerCallback(...)
        callback_eval = nemo.core.EvaluatorCallback(...)
        callback_ckpt = nemo.core.CheckpointCallback(...)

    .. note::

        The BLEU score is calculated between detokenized translation (generated with beam search) and genuine evaluation dataset. For the sake of completeness, we report both  `SacreBLEU <https://github.com/mjpost/sacreBLEU>`_ :cite:`post2018call` and `tokenized BLEU score <https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl>`_ commonly used in the literature.

Finally, we define the optimization parameters and run the whole pipeline.

    .. code-block:: python

        optimizer = neural_factory.get_trainer(**optimization_params)
        optimizer.train(tensors_to_optimize=[train_loss],
                        callbacks=[callback_train, callback_eval, callback_ckpt])


Model training
--------------

To train the Transformer-big model, run ``nmt_tutorial.py`` located at ``nemo/examples/nlp``:

    .. code-block:: python

        python -m torch.distributed.launch --nproc_per_node=<num_gpus> nmt_tutorial.py \
            --data_root <path_to_data> --tokenizer_model bpe8k_yttm.model \
            --eval_datasets valid/newstest2013 --optimizer novograd --lr 0.04 \
            --weight_decay 0.0001 --max_steps 40000 --warmup_steps 4000 \
            --d_model 1024 --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
            --batch_size 12288 --iter_per_step 5


    .. note::

        This command runs training on 8 GPUs with at least 16 GB of memory. If your GPUs have less memory, decrease the **batch_size** parameter. To train with bigger batches which do not fit into the memory, increase the **iter_per_step** parameter.

Translation with pretrained model
---------------------------------

1. Put your saved checkpoint (or download good checkpoint which obtains 28.5 SacreBLEU on newstest2014 from `here <https://drive.google.com/open?id=1ra8yxIKjRGgVM2e2h7PlE4CbGpoKvqnt>`_) into **<path_to_ckpt>**.
2. Run ``nmt_tutorial.py`` in an interactive mode::

    python nmt_tutorial.py --tokenizer_model bpe8k_yttm.model \
         --eval_datasets test --optimizer novograd --d_model 1024 \
         --d_inner 4096 --num_layers 6 --num_attn_heads 16 \
         --checkpoint_dir <path_to_ckpt> --interactive


   .. image:: interactive_translation.png
       :align: center

References
----------

.. bibliography:: nmt.bib
    :style: plain
