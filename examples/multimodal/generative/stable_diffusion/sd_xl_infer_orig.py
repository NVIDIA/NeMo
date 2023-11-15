from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.parts.stable_diffusion.sdxl_helpers import perform_save_locally
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_pipeline import SamplingPipeline
from nemo.core.config import hydra_runner
from nemo.collections.multimodal.models.stable_diffusion.diffusion_engine import DiffusionEngine


'''
SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
'''


@hydra_runner(config_path='conf', config_name='sd_xl_infer')
def main(cfg):
    base_model_config = OmegaConf.load(cfg.base_model_config)
    model = DiffusionEngine(base_model_config.model, {}).cuda()
    base = SamplingPipeline(model, use_fp16=cfg.use_fp16, is_legacy=base_model_config.model.is_legacy)
    use_refiner = cfg.get('use_refiner', False)
    if use_refiner:
        refiner_config = cfg.refiner_config
        refiner = SamplingPipeline(refiner_config, use_fp16=cfg.use_fp16)

    for prompt in cfg.infer.prompt:
        samples = base.text_to_image(
            params=cfg.sampling.base,
            prompt=[prompt],
            negative_prompt=cfg.infer.negative_prompt,
            samples=cfg.infer.num_samples,
            return_latents=True if use_refiner else False,
        )

        if use_refiner:
            assert isinstance(samples, (tuple, list))
            samples, samples_z = samples
            assert samples is not None
            assert samples_z is not None

            perform_save_locally(cfg.out_path, samples)

            samples = refiner.refiner(
                params=cfg.sampling.refiner,
                image=samples_z,
                prompt=cfg.infer.prompt,
                negative_prompt=cfg.infer.negative_prompt,
                samples=cfg.infer.num_samples,
            )

        perform_save_locally(cfg.out_path, samples)


if __name__ == "__main__":
    main()
