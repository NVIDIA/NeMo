from nemo.collections.vlm.grounding_vlm.model import Qwen25VLGroundingConfig3B, Qwen25VLGroundingConfig7B, Qwen25VLGroundingConfig32B, Qwen25VLGroundingConfig72B
from transformers import AutoTokenizer

if __name__ == "__main__":

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    
    # init model config
    model_cfg = Qwen25VLGroundingConfig3B()
    model = model_cfg.configure_model(tokenizer)
    print(model)
