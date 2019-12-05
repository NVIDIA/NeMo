Tutorial
========

Make sure you have ``nemo`` and ``nemo_nlp`` installed before starting this
tutorial. See the :ref:`installation` section for more details.

Introduction
------------

An ASR system typically generates output with no punctuation and capitalization of the words. To make such output more human readable and to boost performance of the downstream tasks such as name entity recoginition or machine translation, this tutorial explains how to implement a model in NeMo that will predict where punctuation and capitalization of a word required. We'll show how to do this with a pre-trained BERT model. For more details on how to pretrain BERT yourself, check out our BERT pretraining tutorial. 

Task Description
----------------
For every word in our training dataset we're going to predict what punctuation mark should follow the word if any and whether the word should be capitalized. The following punctuation marks are considered for this task: commas, periods, question marks. Labels format: ``OL``, ``,L``, ``,U``, ``.L``, ``.U``, ``?L``, ``?U``, ``OU``, where the first symbol of the label indicates the punctuation mark (``O`` - no punctuation needed), and the second symbol determines is the word needs to be upper or lower cased.

.. tip::

    We recommend you try this out in a Jupyter notebook. It'll make debugging much easier!
    See examples/nlp/PunctuationWithBERT.ipynb

Get Data
----------------

For this tutorial, we're going to use the `Tatoeba collection of sentences`_. Use `this`_ script to download and preprocess the dataset. Note that any text dataset will work, the only requirement is that the data is splitted into 2 files: text.txt and labels.txt. The text.txt files should be formatted like this:

.. _Tatoeba collection of sentences: https://tatoeba.org/eng
.. _this: https://github.com/NVIDIA/NeMo/scripts

.. code-block::

    when is the next flight to new york
    the next flight is ...
    ...

The labels.txt files should be formatted like this:

.. code-block::

    OU OL OL OL OL OL OU ?U 
    OU OL OL OL ...
    ...

Each line of the text.txt file contains text sequences, where words are separated with spaces. 
The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces.
Each line of the files should follow the format: 
[WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).

Set parameters for the model
----------------------------
    .. code-block:: python
        DATA_DIR = "PATH_TO_WHERE_THE_DATA_IS"
        WORK_DIR = "PATH_TO_WHERE_TO_STORE_CHECKPOINTS_AND_LOGS"


        # model parameters
        BATCHES_PER_STEP = 1
        BATCH_SIZE = 32
        CLASSIFICATION_DROPOUT = 0.1
        MAX_SEQ_LENGTH = 128
        NUM_EPOCHS = 3
        LEARNING_RATE = 0.00005
        LR_WARMUP_PROPORTION = 0.1
        OPTIMIZER = "adam"
        PRETRAINED_BERT_MODEL = "bert-base-cased"

        # It's import to specify the none_label correctly depending on a task at hand.
        # For combined punctuation and capitalization task use 'OL' for a pucntuation only model the default 'O' will work
        NONE_LABEL = 'OL'
        # determines how often loss will be printed
        STEP_FREQ=200

Training
--------

Examples of training and inference scripts could be found `here`_ and `here`_.
.. _here: https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/token_classification.py
.. _here: https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/token_classification_infer.py

First, we need to create our neural factory with the supported backend. How you should define it depends on whether you'd like to multi-GPU or mixed-precision training. This tutorial assumes that you're training on one GPU, without mixed precision (``optimization_level="O0"``). If you want to use mixed precision, set ``optimization_level`` to ``O1`` or ``O2``.

    .. code-block:: python

        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=None,
                                           optimization_level="O0",
                                           log_dir=WORK_DIR,
                                           placement=nemo.core.DeviceType.GPU)

Next, we'll need to define our tokenizer and our BERT model. If you're using a standard BERT model, you should do it as follows. To see the full list of BERT model names, check out ``nemo_nlp.huggingface.BERT.list_pretrained_models()``

    .. code-block:: python

        tokenizer = NemoBertTokenizer(pretrained_model=PRETRAINED_BERT_MODEL)
        bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name=PRETRAINED_BERT_MODEL)

See examples/nlp/token_classification.py on how to use a BERT model that you pre-trained yourself.
Now, create the train and evaluation data layers:

    .. code-block:: python

        train_data_layer = nemo_nlp.BertTokenClassificationDataLayer(
        tokenizer=tokenizer,
        text_file=os.path.join(DATA_DIR, 'text_train.txt'),
        label_file=os.path.join(DATA_DIR, 'labels_train.txt'),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        pad_label=NONE_LABEL)

        eval_data_layer = nemo_nlp.BertTokenClassificationDataLayer(
        tokenizer=tokenizer,
        text_file=os.path.join(DATA_DIR, 'text_dev.txt'),
        label_file=os.path.join(DATA_DIR, 'labels_dev.txt'),
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        pad_label=NONE_LABEL,
        label_ids=label_ids)


We need to create the classifier to sit on top of the pretrained model and define the loss function:
    .. code-block:: python
    label_ids = train_data_layer.dataset.label_ids
    num_classes = len(label_ids)

    hidden_size = bert_model.local_parameters["hidden_size"]
    classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                              num_classes=num_classes,
                                              dropout=CLASSIFICATION_DROPOUT)

    task_loss = nemo_nlp.TokenClassificationLoss(d_model=hidden_size,
                                            num_classes=len(label_ids),
                                            dropout=CLASSIFICATION_DROPOUT)

Then, create the train and evaluation datasets:

.. code-block:: python
    input_ids, input_type_ids, input_mask, loss_mask, _, labels = train_data_layer()

    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)

    logits = classifier(hidden_states=hidden_states)
    loss = task_loss(logits=logits, labels=labels, loss_mask=loss_mask)

    eval_input_ids, eval_input_type_ids, eval_input_mask, _, eval_subtokens_mask, eval_labels \
        = eval_data_layer()

    hidden_states = bert_model(
        input_ids=eval_input_ids,
        token_type_ids=eval_input_type_ids,
        attention_mask=eval_input_mask)

    eval_logits = classifier(hidden_states=hidden_states)

Now, create the train and evaluation datasets:

.. code-block:: python
    train_tensors, train_loss, steps_per_epoch, label_ids, _ = create_pipeline()
    eval_tensors, _, _, _, data_layer = create_pipeline(mode='dev')

Now, we will set up our callbacks. We will use 3 callbacks:

* `SimpleLossLoggerCallback` to print loss values during training
* `EvaluatorCallback` to evaluate our F1 score on the dev dataset. In this example, `EvaluatorCallback` will also output predictions to `output.txt`, which can be helpful with debugging what our model gets wrong.
* `CheckpointCallback` to save and restore checkpoints.

.. tip::
    
    Tensorboard_ is a great debugging tool. It's not a requirement for this tutorial, but if you'd like to use it, you should install tensorboardX_ and run the following command during fine-tuning:

    .. code-block:: bash
    
        tensorboard --logdir bert_ner_tb

.. _Tensorboard: https://www.tensorflow.org/tensorboard
.. _tensorboardX: https://github.com/lanpa/tensorboardX

    .. code-block:: python

        callback_train = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss],
        print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
        step_freq=STEP_FREQ)

        train_data_size = len(train_data_layer)

        # If you're training on multiple GPUs, this should be
        # train_data_size / (batch_size * batches_per_step * num_gpus)
        steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))

        # Callback to evaluate the model
        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_logits, eval_labels, eval_subtokens_mask],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, label_ids),
            eval_step=steps_per_epoch)

        # Callback to store checkpoints
        ckpt_callback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
            epoch_freq=1)

Finally, we will define our learning rate policy and our optimizer, and start training.

    .. code-block:: python
        lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

        nf.train(tensors_to_optimize=[loss],
                 callbacks=[callback_train, callback_eval, ckpt_callback],
                 lr_policy=lr_policy,
                 batches_per_step=BATCHES_PER_STEP,
                 optimizer=OPTIMIZER,
                 optimization_params={"num_epochs": NUM_EPOCHS,
                                      "lr": LEARNING_RATE})

Training for 3 epochs will take less than 10 mins on a single GPU, expected F1 score is around 0.65.

Inference
---------

To see how the model performs, let's run inference for a few samples. We need to define a data layer for inference the same way we created data layers for training and evaluation.

.. code-block:: python
    queries = ['we bought four shirts from the nvidia gear store in santa clara', 
           'tom sam and i are going to travel do you want to join',
           'nvidia is a company',
           'can i help you',
           'we bought four shirts one mug and ten thousand titan rtx graphics cards the more you buy the more you save']

    # helper functions
    def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])

    def get_preds(logits):
        return np.argmax(logits, 1)

    infer_data_layer = nemo_nlp.BertTokenClassificationInferDataLayer(
                                                queries=queries,
                                                tokenizer=tokenizer,
                                                max_seq_length=MAX_SEQ_LENGTH,
                                                batch_size=1)

Now, run inference and append punctuation and capitalize words based on the generated predictions.

.. code-block:: python

    input_ids, input_type_ids, input_mask, _, subtokens_mask = infer_data_layer()

    hidden_states = bert_model(input_ids=input_ids,
                                          token_type_ids=input_type_ids,
                                          attention_mask=input_mask)
    logits = classifier(hidden_states=hidden_states)

    evaluated_tensors = nf.infer(tensors=[logits, subtokens_mask], checkpoint_dir=WORK_DIR + '/checkpoints')



    ids_to_labels = {label_ids[k]: k for k in label_ids}

    logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

    preds = np.argmax(logits, axis=2)

    for i, query in enumerate(queries):
        nf.logger.info(f'Query: {query}')

        pred = preds[i][subtokens_mask[i] > 0.5]
        words = query.strip().split()
        if len(pred) != len(words):
            raise ValueError('Pred and words must be of the same length')

        output = ''
        for j, word in enumerate(words):
            label = ids_to_labels[pred[j]]
        
            if label != NONE_LABEL:
                if 'U' in label:
                    word = word.capitalize()
                if label[0] != 'O':
                    word += label[0]
                
            output += word
            output += ' '
        nf.logger.info(f'Combined: {output.strip()}\n')

Result for the sample queries should look something like that:
.. code-block:: python

    Query: we bought four shirts from the nvidia gear store in santa clara
    Combined: We bought four shirts from the nvidia gear store in santa clara.

    Query: tom sam and i are going to travel do you want to join
    Combined: Tom Sam, and I are going to travel. Do you want to join?

    Query: nvidia is a company
    Combined: Nvidia is a company.

    Query: can i help you
    Combined: Can I help you?

    Query: we bought four shirts one mug and ten thousand titan rtx graphics cards the more you buy the more you save
    Combined: We bought four shirts, one mug and ten thousand titan, Rtx graphics cards. The more you buy, the more you save.


To train the model with BERT using the provided scripts
-------------------------------------------------------

To run the provided training script:

.. code-block:: bash

    python examples/nlp/token_classification.py --data_dir path/to/data --none_label 'OL' --pretrained_bert_model=bert-base-cased --work_dir output

To run inference:

.. code-block:: bash

    python examples/nlp/token_classification_infer.py --none_label 'OL' --labels_dict path/to/data/label_ids.csv --work_dir output/checkpoints/

Note, label_ids.csv file will be generated during training and stored in the data_dir folder.

Multi GPU Training
------------------

To run training on multiple GPUs, run

.. code-block:: bash
    export NUM_GPUS=2
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/nlp/token_classification.py --num_gpus $NUM_GPUS --none_label 'OL' 
