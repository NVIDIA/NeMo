import_ckpt sys
from nemo.collections import llm      
from nemo.collections.llm import import_ckpt

nemo_model_path  = "/workspace/nemo/models/Llama-3.1-8B-Instruct-Nemo"

if __name__ == '__main__':
    hf_model_path = "/workspace/nemo/models/Llama-3.1-8B-Instruct/"
    import_ckpt(model=llm.LlamaModel(llm.Llama31Config8B()), source=f"hf://{hf_model_path}", output_path=nemo_model_path)
