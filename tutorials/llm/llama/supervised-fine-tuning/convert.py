from nemo.collections import llm
from nemo.collections.llm import Llama2Config7B

if __name__ == "__main__":
    output = llm.import_ckpt(
        model=llm.LlamaModel(config=Llama2Config7B()),
        source="hf:///workspace/Llama-2-7b-hf",
    )

