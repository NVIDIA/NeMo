from nemo.collections.llm import import_ckpt
from nemo.collections import vlm
from nemo.collections.vlm.llama4.model.llama4_omni import Llama4ScoutExperts16Config

if __name__ == '__main__':
    # Specify the Hugging Face model ID
    hf_model_id = "/path/to/llama4_hf_checkpoint"
    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.Llama4OmniModel(Llama4ScoutExperts16Config()),  # Model configuration
        source=f"hf://{hf_model_id}",  # Hugging Face model source
    )
