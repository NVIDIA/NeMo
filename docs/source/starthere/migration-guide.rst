Migration guide to use lightning 2.0
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

* Replace ``trainer.strategy=null`` with ``trainer.strategy=auto`` as `lightning 2.0 doesn't have None strategy <https://lightning.ai/docs/pytorch/stable/common/trainer.html#:~:text=strategy%20(Union%5Bstr%2C%20Strategy%5D)%20%E2%80%93%20Supports%20different%20training%20strategies%20with%20aliases%20as%20well%20custom%20strategies.%20Default%3A%20%22auto%22.>`_

* Remove ``resume_from_checkpoint`` if being used as a trainer flag and pass the path to `Trainer.fit(ckpt_path="...") method <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=used%20Trainer%E2%80%99s%20flag%20resume_from_checkpoint>`_

* Set ``trainer.strategy = "ddp_find_unused_parameters_true"`` if there are unused parameters in your model as lightning 2.0 has find_unused_parameters as False by default
Reference: `NeMo PR 6433 <https://github.com/NVIDIA/NeMo/pull/6433/files#:~:text=Resolve%20conversation-,cfg.trainer.strategy%20%3D%20%22ddp_find_unused_parameters_true%22,-logging.info>`_
More details about this change: `lightning PR 16611 <https://github.com/Lightning-AI/lightning/pull/16611>`_

* If used Trainer's flag ``replace_sampler_ddp`` replace it with `use_distributed_sampler <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=use%20use_distributed_sampler%3B%20the%20sampler%20gets%20created%20not%20only%20for%20the%20DDP%20strategies>`_

* If using ``CheckpointConnector`` replace it with `_CheckpointConnector <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-fbee9218112b5eb07e4b799b868cbe3ab582336157bde6dc7c881daa63965ff5R20>`_

* To set or get ``ckpt_path`` use ``trainer.ckpt_path`` directly instead of calling protected API via ``trainer._checkpoint_connector._ckpt_path`` or using ``trainer._checkpoint_connector.resume_from_checkpoint_fit_path``

* Change ``import load`` from pytorch_lightning.utilities.cloud_io to ``import _load``

* If used ``from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin`` from replace it with `from pytorch_lightning.plugins.precision import MixedPrecisionPlugin <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=used%20the%20pl.plugins.NativeMixedPrecisionPlugin%20plugin>`_ 

*