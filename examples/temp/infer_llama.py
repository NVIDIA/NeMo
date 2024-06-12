from pathlib import Path

from pytorch_lightning import Trainer

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam

if __name__ == '__main__':
    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    trainer = Trainer(  # nl.Trainer(
        devices=1, max_steps=1, accelerator='gpu', strategy=strategy, plugins=nl.MegatronMixedPrecision('bf16-mixed')
    )

    cfg = llm.MegatronGPTModelV2.restore_from('llama3_7b/', trainer=trainer, return_config=True)
    cfg.pop("target", None)

    model = llm.MegatronGPTModelV2.restore_from('llama3_7b/', trainer=trainer, override_config_path=cfg)

    print("Model loaded")
    print("Num parameters: ", model.num_weights)

    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant who answers the question concisely<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    length_params = LengthParam(max_length=32, min_length=1)

    sampling_params = SamplingParam(
        temperature=1.0,
        use_greedy=True,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        add_BOS=False,
        all_probs=False,
        compute_logprob=False,
        end_strings=["</s>", "<|end_of_text|>", "<|eot_id|>"],
    )

    response = model.generate([template.format(input="What can you help me with?")], length_params, sampling_params)
    print(response)
