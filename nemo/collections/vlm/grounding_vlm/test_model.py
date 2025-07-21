from transformers import AutoTokenizer

from nemo.collections.vlm.grounding_vlm.model import Qwen25VLGroundingConfig3B, Qwen2GroundingVLModel

if __name__ == "__main__":

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    
    # init model config
    model_cfg = Qwen25VLGroundingConfig3B()
    model = Qwen2GroundingVLModel(model_cfg, model_version="qwen25-vl", tokenizer=tokenizer)
    model.configure_model()

    print(model.module)
    # model = model_cfg.configure_model(tokenizer)
    # print(model)

