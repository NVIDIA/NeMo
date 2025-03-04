from nemo.collections import llm
from nemo.collections.llm import Llama2Config7B, LlamaModel, import_ckpt

if __name__ == "__main__":
    # # Specify the Hugging Face model ID
    # hf_model_id = "lmsys/vicuna-7b-v1.5"

    # # Import the model and convert to NeMo 2.0 format
    # import_ckpt(
    #     model=LlamaModel(Llama2Config7B()),
    #     source=f"hf://{hf_model_id}",
    # )

    llm.import_ckpt(model=LlamaModel(llm.Llama3Config8B()), source="hf://meta-llama/Meta-Llama-3-8B")
