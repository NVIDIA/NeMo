import math
import time

import numpy as np
import open_clip
import tensorrt as trt
import torch
from cuda import cudart
from transformers import CLIPTokenizer

from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.denoiser import DiscreteDenoiser
from nemo.collections.multimodal.modules.stable_diffusion.quantization_utils.trt_engine import TRT_LOGGER, Engine
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_helpers import perform_save_locally
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_pipeline import get_sampler_config
from nemo.collections.multimodal.parts.stable_diffusion.utils import instantiate_from_config
from nemo.core.classes.common import Serialization
from nemo.core.config import hydra_runner


class StableDiffusionXLTRTPipeline(Serialization):
    def __init__(self, cfg):
        self.modules = ['unet_xl', 'vae', 'clip1', 'clip2']
        self.engine = {}
        self.cfg = cfg

        self.do_classifier_free_guidance = True
        self.denoising_steps = cfg.steps
        self.guidance_scale = cfg.scale
        self.device = cfg.get('device', 'cuda')
        self.in_channels = cfg.get('in_channels', 4)
        self.events = {}
        self.use_cuda_graph = cfg.get('use_cuda_graph', False)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.denoiser = StableDiffusionXLTRTPipeline.from_config_dict(cfg.denoiser_config).to(self.device)
        self.scale_factor = cfg.scale_factor

    def loadEngines(self):
        for model_name in self.modules:
            self.engine[model_name] = Engine(self.cfg.get(f'{model_name}'))
            self.engine[model_name].load()
            print(f"{model_name} trt engine loaded successfully")

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, adm_in_channels, seed):
        # Initialize noise generator
        if seed:
            self.seed = seed
            self.generator = torch.Generator(device="cuda").manual_seed(seed)

        # Create CUDA events and stream
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        for model_name in self.modules:
            self.engine[model_name].allocate_buffers(
                shape_dict=self.get_shape_dict(batch_size, image_height, image_width, adm_in_channels, model_name),
                device=self.device,
            )

    def get_shape_dict(self, batch_size, image_height, image_width, adm_in_channels, model_name):
        if model_name == 'unet_xl':
            feed_dict = {
                'x': (batch_size * 2, 4, image_height // 8, image_width // 8),
                'y': (batch_size * 2, adm_in_channels),
                'timesteps': (batch_size * 2,),
                'context': (batch_size * 2, 80, 2048),
                'out': (batch_size * 2, 4, image_height // 8, image_width // 8),
            }
        elif model_name == 'vae':
            feed_dict = {
                'z': (batch_size, 4, image_height // 8, image_width // 8),
                'dec': (batch_size, 3, image_height, image_width),
            }
        elif model_name == 'clip1':
            feed_dict = {"input_ids": (batch_size, 77), "z": (batch_size, 80, 768)}
        elif model_name == 'clip2':
            feed_dict = {"input_ids": (batch_size, 77), "z": (batch_size, 80, 1280), "z_pooled": (batch_size, 1280)}

        return feed_dict

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def encode_prompt(self, prompt, negative_prompt=""):
        def tokenize(prompt):
            batch_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=77,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
            text_embeddings_1 = self.runEngine('clip1', {"input_ids": tokens})['z']

            tokens = open_clip.tokenize(prompt)
            out_dict = self.runEngine('clip2', {"input_ids": tokens})
            text_embeddings_2, pooled_output = out_dict['z'], out_dict['z_pooled']

            text_embeddings = torch.cat((text_embeddings_1, text_embeddings_2), dim=2)

            return text_embeddings, pooled_output

        c_t, c_pooled = tokenize(prompt)
        uc_t, uc_pooled = tokenize(negative_prompt)

        c, uc = {}, {}
        c['vector'], uc['vector'] = c_pooled, uc_pooled
        c['crossattn'], uc['crossattn'] = c_t, uc_t

        return c, uc

    def run_denoiser_engine(self, x, t, c, **kwargs):
        feed_dict = {}
        feed_dict['x'] = x
        feed_dict['timesteps'] = t
        feed_dict['context'] = c.get("crossattn", None)
        feed_dict['y'] = c.get("vector", None)
        return self.runEngine(model_name="unet_xl", feed_dict=feed_dict)["out"]

    def decode_images(self, samples):
        z = 1.0 / self.scale_factor * samples

        feed_dict = {}
        feed_dict['z'] = z

        return self.runEngine(model_name="vae", feed_dict=feed_dict)["dec"]

    def run(
        self, prompt, negative_prompt, image_height, image_width, num_samples,
    ):

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        prompt = np.repeat(prompt, repeats=math.prod([num_samples])).reshape([num_samples]).tolist()
        negative_prompt = np.repeat(negative_prompt, repeats=math.prod([num_samples])).reshape([num_samples]).tolist()

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)

        self.sampler = get_sampler_config(self.cfg)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            c, uc = self.encode_prompt(prompt, negative_prompt)

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod([num_samples])].to(self.device), (c, uc))

            shape = (math.prod([num_samples]), self.in_channels, latent_height, latent_width)
            randn = torch.randn(shape, generator=self.generator, device=self.device)

            def denoiser(input, sigma, c):
                additional_model_inputs = {}
                return self.denoiser(self.run_denoiser_engine, input, sigma, c, **additional_model_inputs)

            samples_z = self.sampler(denoiser, randn, cond=c, uc=uc)
            samples_x = self.decode_images(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            perform_save_locally(self.cfg.out_path, samples)


@hydra_runner(config_path='conf', config_name='sd_xl_trt_inference')
def main(cfg):
    base = StableDiffusionXLTRTPipeline(cfg)

    base.loadEngines()

    base.activateEngines()

    base.loadResources(cfg.height, cfg.width, cfg.num_samples, cfg.adm_in_channels, cfg.seed)

    for prompt in cfg.prompts:
        base.run([prompt], "", cfg.width, cfg.height, cfg.num_samples)


if __name__ == "__main__":
    main()
