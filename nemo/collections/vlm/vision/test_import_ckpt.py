from nemo.collections.llm import import_ckpt
from nemo.collections.vlm.vision.clip_vit import CLIPViTModel
from nemo.collections.vlm.vision.vit_config import CLIPViTL_14_336_Config

if __name__ == "__main__":
    # Specify the Hugging Face model ID
    hf_model_id = "openai/clip-vit-large-patch14-336"

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=CLIPViTModel(CLIPViTL_14_336_Config()),
        source=f"hf://{hf_model_id}",
    )
