Tutorial
===========================

In this tutorial we will train an ASR postprocessing model to correct mistakes in
output of end-to-end speech recognition model. This model method works similar to translation model in contrast to traditional ASR language model rescoring.
The model architecture is attention based encoder-decoder where both encoder and decoder are initialized with pretrained BERT language model.
To train this model we collected dataset with typical ASR errors by using pretrained Jasper ASR model :cite:`asr-imps-li2019jasper`.

Data
-----------
**Data collection.** We collected dataset for this tutorial with Jasper ASR model
:cite:`asr-imps-li2019jasper` trained on Librispeech dataset :cite:`asr-imps-panayotov2015librispeech`.
To download the Librispeech dataset, see :ref:`LibriSpeech_dataset`.
To obtain the pretrained Jasper model, see :ref:`Jasper_model`.
Librispeech training dataset consists of three parts: train-clean-100, train-clean-360, and train-clean-500 which give 281k training examples in total.
To augment this data we used two techniques:

* We split all training data into 10 folds and trained 10 Jasper models in cross-validation manner: a model was trained on 9 folds and used to make ASR predictions for the remaining fold.

* We took pretrained Jasper model and enabled dropout during inference on training data. This procedure was repeated multiple times with different random seeds.

**Data postprocessing.** The collected dataset was postprocessed by removing duplicates
and examples with word error rate higher than 0.5.
The resulting training dataset consists of 1.7M pairs of "bad" English-"good" English examples.

**Dev and test datasets preparation**. Librispeech contains 2 dev datasets
(dev-clean and dev-other) and 2 test datasets (test-clean and test-other).
For our task we kept the same splits. We fed these datasets to a pretrained
Jasper model with the greedy decoding to get the ASR predictions that are used
for evaluation in our tutorial.

Importing parameters from pretrained BERT
-----------------------------------------
Both encoder and decoder are initialized with pretrained BERT parameters.
Since BERT language model has the same architecture as transformer encoder, there is no need to do anything additional.
To prepare decoder parameters from pretrained BERT we wrote a script ``get_decoder_params_from_bert.py`` that downloads BERT
parameters from the ``pytorch-transformers`` repository :cite:`asr-imps-huggingface2019transformers` and maps them into a transformer decoder.
Encoder-decoder attention is initialized with self-attention parameters.
The script is located under ``scripts`` directory and accepts 2 arguments:

* ``--model_name``: e.g. ``bert-base-cased``, ``bert-base-uncased``, etc.
* ``--save_to``: a directory where the parameters will be saved

    .. code-block:: bash

        $ python get_decoder_params_from_bert.py --model_name bert-base-uncased


Neural modules overview
--------------------------
First, as with all models built in NeMo, we instantiate Neural Module Factory which defines 1) backend (PyTorch or TensorFlow), 2) mixed precision optimization level, 3)
local rank of the GPU, and 4) an experiment manager that creates a timestamped folder to store checkpoints, relevant outputs, log files, and TensorBoard graphs.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(
                        backend=nemo.core.Backend.PyTorch,
                        local_rank=args.local_rank,
                        optimization_level=args.amp_opt_level,
                        log_dir=work_dir,
                        create_tb_writer=True,
                        files_to_copy=[__file__])


Then we define tokenizer to convert tokens into indices. We will use ``bert-base-uncased`` vocabulary, since our dataset only contains uncased text:

    .. code-block:: python

        tokenizer = NemoBertTokenizer(pretrained_model="bert-base-uncased")


The encoder block is a neural module corresponding to BERT language model from
``nemo_nlp.huggingface`` collection:

    .. code-block:: python

        zeros_transform = nemo.backends.pytorch.common.ZerosLikeNM()
        encoder = nemo_nlp.huggingface.BERT(
            pretrained_model_name=args.pretrained_model,
            local_rank=args.local_rank)

    .. tip::
        Making embedding size (as well as all other tensor dimensions) divisible
        by 8 will help to get the best GPU utilization and speed-up with mixed precision training.

    .. code-block:: python

        vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
        tokens_to_add = vocab_size - tokenizer.vocab_size
        
        device = encoder.bert.embeddings.word_embeddings.weight.get_device()
        zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)

        encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
            (encoder.bert.embeddings.word_embeddings.weight.data, zeros))


Next, we construct transformer decoder neural module. Since we will be initializing decoder with pretrained BERT parameters, we set hidden activation to ``"hidden_act": "gelu"`` and
learn positional encodings ``"learn_positional_encodings": True``:

    .. code-block:: python

        decoder = nemo_nlp.TransformerDecoderNM(
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

To load the pretrained parameters into decoder, we use ``restore_from`` attribute function of the decoder neural module:

    .. code-block:: python

        decoder.restore_from(args.restore_from, local_rank=args.local_rank)


Model training
--------------

To train the model run ``asr_postprocessor.py.py`` located in ``examples/nlp`` directory. We train with novograd optimizer :cite:`asr-imps-ginsburg2019stochastic`,
learning rate ``lr=0.001``, polynomial learning rate decay policy, ``1000`` warmup steps, per-gpu batch size of ``4096*8`` tokens, and ``0.25`` dropout probability.
We trained on 8 GPUS. To launch the training in multi-gpu mode run the following command:

    .. code-block:: bash

        $ python -m torch.distributed.launch --nproc_per_node=8  asr_postprocessor.py --data_dir ../../tests/data/pred_real/ --restore_from ../../scripts/bert-base-uncased_decoder.pt



References
------------------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: ASR-IMPROVEMENTS
    :keyprefix: asr-imps-    
