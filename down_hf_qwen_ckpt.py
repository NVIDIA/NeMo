from nemo.collections.llm import import_ckpt
from nemo.collections import vlm

if __name__ == '__main__':
  # Specify the Hugging Face model ID
    hf_model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.Qwen2VLModel(vlm.Qwen2VLConfig2B()),  # Model configuration
        source=f"hf://{hf_model_id}",  # Hugging Face model source
    )