inference_params: InferenceParams = {
       "use_greedy": True,
       "temperature": 1.0,
       "top_k": 0,
       "top_p": 1.0,
       "repetition_penalty": 1.0,
       "add_BOS": True,
       "all_probs": False,
       "compute_logprob": False,
       "end_strings": ["<|endoftext|>", "<extra_id_1>"],
	    "min_length": 0, 
        "max_length": 30
   }


class MCoreBackend():
    def __init__(self, model, tokenizer, text_generation_strategy = None):
        self.text_gen_strategy = None
        if text_generation_strategy is None:
            self.text_gen_strategy = SimpleTextGenerationStrategy(model, inference_params, tokenizer)
        else :
            self.text_gen_strategy = text_generation_strategy

    def generate(self, prompts, inference_params):  
        inp_tokens = self.text_gen_strategy.tokenizer_batch(prompts)
        logits = self.text_gen_strategy.forward_step(inp_tokens)
        output = self.text_gen_strategy.post_process_generations(logits)
        return output
        

class SimpleTextGenerationStrategy():
   	def init(self, model, inference_params, tokenizer):
        return
		
	def forward_step(self, batch):
        return

	def tokenize_batch(self, prompts):
        return

	def post_process_generations(self, logits, post_process_fn):
        # detokenize , #beam search etc ? 
        return post_process_fn(tokens)
 


def model_specific_generate(inference_params, prompts, backend = None):
    if backend is None:
        if  model is trt_llm_exportable : 
            backend = TrtLLM
        else :
            backend = MCoreBackend(model, tokenizer, SimpleTextGenerationStrategy)   

    common_generate(inference_params, backend, prompts)


def common_generate(inference_params, backend, prompts):
    backend.generate(prompts, inference_params)
        
