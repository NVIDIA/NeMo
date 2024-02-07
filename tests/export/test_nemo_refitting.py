# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
import shutil
import os
import torch
from mpi4py import MPI
        
import omegaconf
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.export import TensorRTLLM
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector

from tests.infer_data_path import get_infer_test_data
from tests.export.test_nemo_export import get_accuracy_with_lambada, get_args

class RefitTester:
    def _init_nemo_inference(self, model_info, tp, pp, load_model=True):
        
        import nemo

        sampling_config = os.path.join(
            os.path.dirname(os.path.dirname(nemo.__file__)),
            'examples/nlp/language_modeling/conf/megatron_gpt_inference.yaml',
        )

        cfg = omegaconf.OmegaConf.load(sampling_config)
        cfg.gpt_model_file = model_info["checkpoint"]

        # trainer required for restoring model parallel models
        trainer = Trainer(strategy=NLPDDPStrategy(), devices=MPI.COMM_WORLD.Get_size())

        gpt_model_file = model_info["checkpoint"]
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(gpt_model_file):
            save_restore_connector.model_extracted_dir = gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=gpt_model_file,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.micro_batch_size = 1
        pretrained_cfg.tensor_model_parallel_size = tp
        pretrained_cfg.pipeline_model_parallel_size = pp
        
        if load_model:
            model = MegatronGPTModel.restore_from(
                restore_path=gpt_model_file,
                trainer=trainer,
                override_config_path=pretrained_cfg,
                save_restore_connector=save_restore_connector)
            model.freeze()
        else:
            model = None
        
        trainer.strategy.setup_distributed()

        self.nemo_model = model
        self.nemo_inference_cfg = cfg
        self.nemo_model_cfg = pretrained_cfg
        if self.nemo_model_cfg.get('transformer_engine', False) or self.nemo_model_cfg.get('mcore_gpt', False):
            model.setup_transformer_engine_tp_groups()

    def _config(self):
        self.nemo_inference_cfg.inference.min_tokens_to_generate = 200
        self.nemo_inference_cfg.inference.tokens_to_generate = 200
        self.nemo_inference_cfg.inference.top_k = 1
        self.nemo_inference_cfg.inference.top_p = 0
        self.nemo_inference_cfg.inference.temperature = 0.1
        self.nemo_inference_cfg.inference.greedy = True
        self.length_params: LengthParam = {
            "max_length": self.nemo_inference_cfg.inference.tokens_to_generate,
            "min_length": self.nemo_inference_cfg.inference.min_tokens_to_generate,
        }

        self.sampling_params: SamplingParam = {
            "use_greedy": self.nemo_inference_cfg.inference.greedy,
            "top_k": self.nemo_inference_cfg.inference.top_k,
            "top_p": self.nemo_inference_cfg.inference.top_p,
            "temperature": self.nemo_inference_cfg.inference.temperature,
            "repetition_penalty": self.nemo_inference_cfg.inference.repetition_penalty,
            "all_probs": False,
            "compute_logprob": self.nemo_inference_cfg.inference.compute_logprob,
            "end_strings": ['<|endoftext|>'],
            "max_length": 100
        }
        
        self.trt_llm_inference_params ={
            "top_k": self.sampling_params.get("top_k", None),
            "top_p": self.sampling_params.get("top_p", None),
            "temperature": self.sampling_params.get("temperature", None),
            "output_log_probs": True,
            "max_output_token": self.sampling_params.get("max_length", None),
        }
    
    def run(self, model_name, tp_size, pp_size, run_accuracy=False):
        test_data = get_infer_test_data()
        model_info = test_data[model_name]
            
        try: 
            Path(model_info["checkpoint"]).exists()
        except FileNotFoundError:
            print("Checkpoint {0} could not be found.".format(model_info["checkpoint"]))
    
        Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"Path: {0} and model: {1} will be tested".format(
            model_info["checkpoint"], model_name)
        )
        
        # Initialize NeMo model
        self._init_nemo_inference(model_info, tp_size, pp_size)
        self._config()
        
        # Build engine from nemo
        trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"], load_model=False)
        trt_llm_exporter.build(
            self.nemo_model,
            self.nemo_model_cfg, 
            self.nemo_model.tokenizer,
            max_input_token=1024,
            max_output_token=128,
            max_batch_size=model_info["max_batch_size"],
            use_refit=True, 
            model_type=model_info['model_type'])
        
        # Delete, reload, and refit a new TRTLLM model from nemo
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        del tensorrt_llm_worker_context.decoder
            
        trt_llm_exporter.refit(self.nemo_model, self.nemo_model_cfg)
        
        output, _ = trt_llm_exporter.forward(
            input_texts=model_info["prompt_template"], 
            **self.trt_llm_inference_params
        )
        print(f"Rank {torch.distributed.get_rank()} refit output: {output}")
        if run_accuracy:
            trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs = get_accuracy_with_lambada(
                trt_llm_exporter, None, None
            )
            print("Model Accuracy: {0}, Relaxed Model Accuracy: {1}".format(trtllm_accuracy, trtllm_accuracy_relaxed))
            assert trtllm_accuracy_relaxed > 0.5, "Model accuracy is below 0.5"
        
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(model_info["trt_llm_model_dir"])

if __name__ == '__main__':
    args = get_args()

    refit_runner = RefitTester()
    refit_runner.run(
        args.model_name, 
        args.tp_size, 
        args.pp_size,
        args.run_accuracy) 

