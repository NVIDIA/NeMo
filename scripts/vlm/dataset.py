from nemo.collections.llm import import_ckpt
from nemo.collections import vlm
from nemo.collections.vlm import ClipConfigL14

if __name__ == '__main__':
    # Specify the Hugging Face model ID
    hf_model_id = "hf://openai/clip-vit-large-patch14"

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.CLIPModel(ClipConfigL14()),  # Model configuration
        source=f"{hf_model_id}",  # Hugging Face model source
    )