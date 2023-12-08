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

import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorrt_llm
import torch
from pytriton.decorators import batch
from pytriton.model_config import Tensor
import tempfile

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list
from nemo.utils import logging

from .trt_llm.model_config_trt import model_config_to_tensorrt_llm
from .trt_llm.nemo_utils import get_tokenzier, nemo_to_model_config
from .trt_llm.tensorrt_llm_run import generate, load
from .utils import is_nemo_file, unpack_nemo_ckpt


class TensorRTLLM(ITritonDeployable):

    """
    Exports nemo checkpoints to TensorRT-LLM and run fast inference.

    Example:
        from nemo.export import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            n_gpus=1,
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)

    """

    def __init__(self, model_dir: str, load_model: bool=True):
        """
        Args:
            model_dir (str): path for storing the TensorRT-LLM model files.
            load_model (bool): load TensorRT-LLM model if the engine files exist in the model_dir.
        """

        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.prompt_table = None
        self.task_vocab_size = None
        self.n_gpus = None
        self.config = None
        if load_model:
            self.load()

    def load(self):
        self.model = None
        self.tokenizer = None
        self.prompt_table = None
        self.task_vocab_size = None
        self.n_gpus = None
        self.config = None

        if Path(self.model_dir).exists():
            folders = os.listdir(self.model_dir)
            if len(folders) > 0:
                try:
                    self._load_config_file()
                    self.tokenizer = get_tokenzier(Path(os.path.join(self.model_dir)))
                    self.model = load(tokenizer=self.tokenizer, engine_dir=self.model_dir)
                    self._load_prompt_table()
                except Exception as error:
                    raise Exception(
                        "Files in the TensorRT-LLM folder is corrupted and "
                        "model needs to be exported again. "
                        "Error message: " + str(error)
                    )

    def _load_prompt_table(self):
        path = Path(os.path.join(self.model_dir, "__prompt_embeddings__.npy"))
        if path.exists():
            self.prompt_table = torch.from_numpy(np.load(path))
            self.task_vocab_size = 1

            dtype = self.config['builder_config']['precision']
            self.prompt_table = self.prompt_table.cuda().to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

            if self.prompt_table.shape[1] != self.config["builder_config"]["hidden_size"]:
                raise Exception(
                    "Hidden dimension of the model is {0} and does not match with the dimension of the prompt table.".format(self.config["builder_config"]["hidden_size"])
                )
        else:
            self.prompt_table = None
            self.task_vocab_size = None

    def _load_config_file(self):
        engine_dir = Path(self.model_dir)
        config_path = engine_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError("file: {0} could not be found.".format(config_path))

    def _get_prompt_embedding_table_ckpt(self, prompt_embeddings_checkpoint_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            unpack_nemo_ckpt(prompt_embeddings_checkpoint_path, temp_dir)
            mw_path = os.path.join(temp_dir, "model_weights.ckpt")
            if not Path(mw_path).exists():
                mw_path = os.path.join(temp_dir, "mp_rank_00", "model_weights.ckpt")
                if not Path(mw_path).exists():
                    raise FileNotFoundError("File: {0} could not be found in the nemo checkpoint. "
                                            "Please check the nemo checkpoint format for the prompt "
                                            "embedding table.".format(mw_path))
            weights = torch.load(mw_path)
            weights = weights["model.embedding.adapter_layer.ptuning_adapter.inference_table"]

            return weights.cpu().detach().numpy()

        return None

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: str,
        prompt_embeddings_table = None,
        prompt_embeddings_checkpoint_path=None,
        delete_existing_files: bool = True,
        n_gpus: int = 1,
        tensor_parallel_size = None,
        pipeline_parallel_size = None,
        max_input_token: int = 256,
        max_output_token: int = 256,
        max_batch_size: int = 8,
        use_inflight_batching: bool = False,
        enable_context_fmha: bool = True,
        paged_kv_cache: bool = False,
        dtype: str = "bfloat16",
        load_model: bool = True,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            nemo_checkpoint_path (str): path for the nemo checkpoint.
            model_type (str): type of the model. Currently, "llama", "gptnext", "falcon", and "starcoder" are supported.
            prompt_embeddings_table: prompt embeddings table.
            prompt_embeddings_checkpoint_path (str): path for the nemo checkpoint for the prompt embedding table.
            delete_existing_files (bool): if Truen, deletes all the files in model_dir.
            n_gpus (int): number of GPUs to use for inference.
            tensor_parallel_size (int): tensor parallelism.
            pipeline_parallel_size (int): pipeline parallelism.
            max_input_token (int): max input length.
            max_output_token (int): max output length.
            max_batch_size (int): max batch size.
            use_inflight_batching (bool): if True, enables inflight batching for TensorRT-LLM Triton backend.
            enable_context_fmha (bool): if True, use fused Context MultiHeadedAttention.
            paged_kv_cache (bool): if True, uses kv cache feature of the TensorRT-LLM.
            dtype (str): Floating point type for model weights (Supports BFloat16/Float16).
            load_model (bool): load TensorRT-LLM model after the export.
        """

        if not model_type in self.get_supported_models_list:
            raise Exception("Model {0} is not currently a supported model type. "
                            "Supported model types are llama, gptnext, falcon, and starcoder".format(model_type))

        if model_type == "gpt" or "starcoder":
            # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
            # gpt and starcoder use the similar model architecture. So, gpt can be used for starcoder.
            model_type = "gptnext"

        if pipeline_parallel_size is None:
            tensor_parallel_size = n_gpus
            pipeline_parallel_size = 1
        elif tensor_parallel_size is None:
            tensor_parallel_size = 1
            pipeline_parallel_size = n_gpus

        p_tuning = "no_ptuning"
        if prompt_embeddings_table is not None and prompt_embeddings_checkpoint_path is not None:
            logging.warning("prompt_embeddings_table will be used and "
                            "prompt_embeddings_checkpoint_path will be "
                            "ignored for ptuning.")
            p_tuning = "use_table"
        elif prompt_embeddings_table is not None:
            p_tuning = "use_table"
        elif prompt_embeddings_checkpoint_path is not None:
            p_tuning = "use_checkpoint"

        if p_tuning == "use_table":
            if not isinstance(prompt_embeddings_table, np.ndarray):
                raise TypeError("Only numpy array is allowed for the prompt embeddings table.")

            if len(prompt_embeddings_table.shape) != 2:
                raise Exception("A two dimensional prompt embeddings table for a single task is only supported.")
        elif p_tuning == "use_checkpoint":
            if not is_nemo_file(prompt_embeddings_checkpoint_path):
                raise TypeError(prompt_embeddings_checkpoint_path + " is not a nemo file.")
            prompt_embeddings_table = self._get_prompt_embedding_table_ckpt(prompt_embeddings_checkpoint_path)

        if prompt_embeddings_table is None:
            max_prompt_embedding_table_size = 0
        else:
            max_prompt_embedding_table_size = len(prompt_embeddings_table)

        if Path(self.model_dir).exists():
            if delete_existing_files and len(os.listdir(self.model_dir)) > 0:
                for files in os.listdir(self.model_dir):
                    path = os.path.join(self.model_dir, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)

                if len(os.listdir(self.model_dir)) > 0:
                    raise Exception("Couldn't delete all files.")
            elif len(os.listdir(self.model_dir)) > 0:
                raise Exception("There are files in this folder. Try setting delete_existing_files=True.")
        else:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.model = None

        tmp_dir = tempfile.TemporaryDirectory()
        nemo_export_dir = Path(tmp_dir.name)

        model_configs, self.tokenizer = nemo_to_model_config(
            in_file=nemo_checkpoint_path,
            decoder_type=model_type,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            nemo_export_dir=nemo_export_dir,
        )

        model_config_to_tensorrt_llm(
            model_configs,
            self.model_dir,
            world_size=tensor_parallel_size*pipeline_parallel_size,
            max_input_len=max_input_token,
            max_output_len=max_output_token,
            max_batch_size=max_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            use_inflight_batching=use_inflight_batching,
            paged_kv_cache=paged_kv_cache,
            enable_context_fmha=enable_context_fmha,
        )

        if p_tuning != "no_ptuning":
            np.save(os.path.join(self.model_dir, "__prompt_embeddings__.npy"), prompt_embeddings_table)

        tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            shutil.copy(tokenizer_path, self.model_dir)
        else:
            self.tokenizer.save_pretrained(os.path.join(self.model_dir, 'huggingface_tokenizer'))
        tmp_dir.cleanup()

        if load_model:
            self.load()

    def forward(
        self,
        input_texts,
        max_output_token=512,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        **sampling_kwargs,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            sampling_kwargs: Additional kwargs to set in the SamplingConfig.
        """
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )
        else:
            return generate(
                input_texts=input_texts,
                max_output_len=max_output_token,
                host_context=self.model,
                top_k= top_k,
                top_p= top_p,
                temperature= temperature,
                prompt_table=self.prompt_table,
                task_vocab_size=self.task_vocab_size,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=False,
                **sampling_kwargs,
            )

    @property
    def get_supported_models_list(self):
        # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
        return ["gpt", "gptnext", "llama", "falcon", "starcoder"]

    def get_hidden_size(self):
        if self.config is None:
            return None
        else:
            return self.config["builder_config"]["hidden_size"]

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(1,), dtype=bytes),
            Tensor(name="max_output_token", shape=(1,), dtype=np.int_),
            Tensor(name="top_k", shape=(1,), dtype=np.int_),
            Tensor(name="top_p", shape=(1,), dtype=np.single),
            Tensor(name="temperature", shape=(1,), dtype=np.single),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            input_texts = str_ndarray2list(inputs.pop("prompts"))
            max_output_token = inputs.pop("max_output_token")
            top_k = inputs.pop("top_k")
            top_p = inputs.pop("top_p")
            temperature = inputs.pop("temperature")

            output_texts = self.forward(
                input_texts=input_texts,
                max_output_token=max_output_token[0][0],
                top_k=top_k[0][0],
                top_p=top_p[0][0],
                temperature=temperature[0][0],
            )

            output = cast_output(output_texts, np.bytes_)
            return {"outputs": output}
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)
            return {"outputs": output}
