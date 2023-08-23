Hidden Transformations
======================

The hidden transformations model extension allows to add hidden transformations and losses to Megatron encoder-decoder models.
Hidden transformations are transformations that are applied to the output of the encoder.
Hidden losses are losses that are applied to the outputs of the hidden transformations.
A common use case for hidden transformations is to train a Mutual Information Machine (MIM)
or a Variational Auto-Encoder (VAE) models.

Quick Start
-----------

Below is an example of how to train a MIM model with BART data augmentation (i.e., includes masking the input).

.. code-block:: bash

    python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=/results/megatron_mim \
        model.micro_batch_size=2 \
        model.global_batch_size=4 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.arch=perceiver \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.data.data_impl=text_mmap \
        model.data.data_prefix=[1.0,/data/wiki.txt] \
        model.data.splits_string=\'\"800,100,100\"\' \
        model.data.whole_word_masking=False \
        model.tokenizer.library=sentencepiece \
        model.tokenizer.model=/data/spm_64k_all_langs_plus_en.model \
        ++model.hiddens.enc_output_name=z \
        ++model.hiddens.transform.q_z_given_x.cls_name=cond_gaussian \
        ++model.hiddens.transform.q_z_given_x.hidden_size=64 \
        ++model.hiddens.loss.mim.cls_name=a_mim \
        ++model.hiddens.loss.mim.loss_weight=1.0

The last 5 lines in the above command enable sampling with reparametrization (`cond_gauss` hidden transformation)
and MIM loss with hidden part of the loss is weighted by `1.0`.
See below detailed description of extension and the configuration format.

Hiddens Extension Detailed Description
--------------------------------------

Hidden Transformations and losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Megatron encoder-decoder models directly passes the outputs of the encoder to the decoder.
The hidden transformations provides a mechanism to add transformations and losses to the encoder outputs.
This is achieved my naming the outputs of the encoder (`hiddens`) and any provided hidden transformation.
This also allows to define losses on any of the named outputs (i.e., the outputs of the encoder or any of the transformations).

Listing and Registering Hidden Transformations and Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nemo.collections.nlp.modules.common.hiddens import (
        get_registered_hiddens, 
        register_hidden_loss, 
        register_hidden_transform,
        get_hiddens_module,
        MegatronBaseHiddenTransform,
        MegatronBaseHiddenLoss,
    )

    # List all registered hidden transformations and losses
    
