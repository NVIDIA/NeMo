import nemo.collections.diffusion
from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
from nemo.collections.diffusion.models.flux.model import ClipConfig, FluxConfig, FluxModelParams, T5Config
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNetConfig, MegatronFluxControlNetModel
from nemo.collections.diffusion.vae.autoencoder import AutoEncoderConfig
from nemo.collections.llm.recipes.log.default import default_resume, tensorboard_logger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig


from nemo.collections import llm
import nemo_run as run
from nemo import lightning as nl
import lightning.pytorch as pl
import torch

NAME='flux_controlnet_training_test'

@run.cli.factory
@run.autoconvert
def flux_mock_datamodule() -> pl.LightningDataModule:
    """Mock Datamodule Initialization"""
    data_module = MockDataModule(
        image_h=1024,
        image_w=1024,
        micro_batch_size=1,
        global_batch_size=1,
        image_precached=True,
        text_precached=True,
    )
    return data_module


@run.cli.factory(target=llm.train, name=NAME)
def flux_controlnet_training(
    flux_num_joint_layers=1,
    flux_num_single_layers=1,
    flux_controlnet_num_joint_layers=1,
    flux_controlnet_num_single_layers=1,
) -> run.Partial:
    """Flux Controlnet Training Config"""
    return run.Partial(
        llm.train,
        model=run.Config(
            MegatronFluxControlNetModel,
            flux_params=run.Config(
                FluxModelParams,
                flux_config=run.Config(
                    FluxConfig,
                    num_joint_layers=flux_num_joint_layers,
                    num_single_layers=flux_num_single_layers,
                ),
                t5_params=None,
                clip_params = None,
                vae_config = None,
            ),
            flux_controlnet_config=run.Config(
                FluxControlNetConfig,
                num_joint_layers=flux_controlnet_num_joint_layers,
                num_single_layers=flux_controlnet_num_single_layers,
            ),
        ),
        data=flux_mock_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices=1,
            num_nodes=1,
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                ),
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=0,
            max_steps=10,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    nl.ModelCheckpoint,
                    save_last=False,
                )
            ]
        ),
        log=run.Config(
            nl.NeMoLogger,
            ckpt=None,
            name=NAME,
            tensorboard=tensorboard_logger(name=NAME),
            log_dir=None,
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                adam_beta1=0.9,
                adam_beta2=0.999,
                use_distributed_optimizer=True,
                bf16=True,
            ),
        ),
        resume=default_resume(),
    )


if __name__ == "__main__":
    recipe = flux_controlnet_training()

    run.cli.main(llm.train, default_factory=flux_controlnet_training)