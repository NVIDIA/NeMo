from torch import nn
from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder
from nemo.collections.diffusion.flux.model import Flux
from nemo.collections.diffusion.schedulers.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.mcore_parallel_utils import Utils
from nemo.collections.diffusion.utils.flux_pipeline_utils import configs
from nemo.collections.diffusion.flux.pipeline import FluxInferencePipeline


if __name__ == '__main__':
    Utils.initialize_distributed(1,1,1)
    params = configs['flux']

    pipe = FluxInferencePipeline(params)
    import pdb; pdb.set_trace()
    text = ['a cat']
    pipe(text, max_sequence_length=256)

