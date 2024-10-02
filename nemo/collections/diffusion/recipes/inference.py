from torch import nn
from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder
from nemo.collections.diffusion.flux.model import Flux
from nemo.collections.diffusion.schedulers.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.mcore_parallel_utils import Utils
from nemo.collections.diffusion.utils.flux_pipeline_utils import configs
from nemo.collections.diffusion.flux.pipeline import FluxInferencePipeline
from nemo.collections.diffusion.recipes.ckpt_converter import flux_transformer_converter
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors


if __name__ == '__main__':
    Utils.initialize_distributed(1,1,1)
    params = configs['flux']

    pipe = FluxInferencePipeline(params)

    ckpt = flux_transformer_converter('/ckpts/transformer', transformer_config=pipe.transformer.transformer_config)

    save_safetensors(ckpt ,"nemo_flux.safetensors")

    missing, unexpected = pipe.transformer.load_state_dict(ckpt, strict=False)
    print('missing', missing)
    text = ['a cat']
    pipe(text, max_sequence_length=256, height=1024, width=1024, num_inference_steps=4,num_images_per_prompt=1, offload=True)

