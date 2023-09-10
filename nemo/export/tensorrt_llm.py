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

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list

from .trt_llm.model_config_trt import model_config_to_tensorrt_llm
from .trt_llm.nemo_utils import get_tokenzier, nemo_to_model_config
from .trt_llm.quantization_utils import naive_quantization
from .trt_llm.tensorrt_llm_run import generate, load
from .utils import get_prompt_embedding_table, is_nemo_file, torch_to_numpy


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

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir (str): path for storing the TensorRT-LLM model files.
        """

        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.prompt_table = None
        self.task_vocab_size = None
        self.n_gpus = None
        self.config = None
        self._load()

    def _load(self):
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
                except:
                    raise Exception(
                        "Files in the TensorRT-LLM folder is corrupted and model needs to be exported again."
                    )

    def _load_prompt_table(self):
        path = Path(os.path.join(self.model_dir, "__prompt_embeddings__.npy"))
        if path.exists():
            self.prompt_table = torch.from_numpy(np.load(path))
            if len(self.prompt_table.shape) > 2:
                self.task_vocab_size = self.prompt_table.shape[1]
                self.prompt_table = self.prompt_table.view(
                    (self.prompt_table.shape[0] * self.prompt_table.shape[1], self.prompt_table.shape[2])
                )
            else:
                self.task_vocab_size = 1

            dtype = self.config['builder_config']['precision']
            self.prompt_table = self.prompt_table.cuda().to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

            if self.prompt_table.shape[1] != self.config["builder_config"]["hidden_size"]:
                raise Exception(
                    "Dimension of the prompt table and the hidden size do not match. Please make sure to use the correct prompt table."
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

    def _save_config_file(self):
        engine_dir = Path(self.model_dir)
        config_path = engine_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        

    def _extract_prompt_embeddings(self, prompt_checkpoint_path):
        if is_nemo_file(prompt_checkpoint_path):
            vtokens_embeddings = get_prompt_embedding_table(prompt_checkpoint_path)
            np.save(os.path.join(self.model_dir, "__prompt_embeddings__.npy"), torch_to_numpy(vtokens_embeddings))
        else:
            raise FileNotFoundError("file: {0} does not exist or is not a nemo file.".format(prompt_checkpoint_path))

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: str,
        prompt_checkpoint_path: str = None,
        delete_existing_files: bool = True,
        n_gpus: int = 1,
        max_input_token: int = 512,
        max_output_token: int = 512,
        max_prompt_embedding_table_size: int = 100,
        max_batch_size: int = 32,
        quantization: bool = None,
        parallel_build: bool = False,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            nemo_checkpoint_path (str): path for the nemo checkpoint.
            model_type (str): type of the model. Currently supports "llama" and "gptnext".
            prompt_checkpoint_path (str): path for the nemo checkpoint that includes prompt table.
            delete_existing_files (bool): if Truen, deletes all the files in model_dir.
            n_gpus (int): number of GPUs to use for inference.
            max_input_token (int): max input length.
            max_output_token (int): max output length.
            max_prompt_embedding_table_size (int): max prompt embedding table size.
            max_batch_size (int): max batch size.
            quantization (bool): if True, applies naive quantization.
            parallel_build (bool): build in parallel or not.
        """

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

        nemo_export_dir = os.path.join(self.model_dir, "/tmp_nemo/")
        model_configs, self.tokenizer = nemo_to_model_config(
            in_file=nemo_checkpoint_path, decoder_type=model_type, gpus=n_gpus, nemo_export_dir=nemo_export_dir
        )

        if prompt_checkpoint_path is None:
            max_prompt_embedding_table_size = 0

        model_config_to_tensorrt_llm(
            model_configs,
            self.model_dir,
            n_gpus,
            max_input_len=max_input_token,
            max_output_len=max_output_token,
            max_batch_size=max_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )

        if prompt_checkpoint_path is not None:
            self._extract_prompt_embeddings(prompt_checkpoint_path)

        shutil.copy(os.path.join(nemo_export_dir, "tokenizer.model"), self.model_dir)
        shutil.rmtree(nemo_export_dir)
        self._load()

    def forward(
        self,
        input_texts,
        max_output_token=512,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
        """
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported and " "TensorRT LLM should be loaded first to run inference."
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
            )

    def set_prompt_embeddings(self, prompt_embeddings):
        """
        Adds prompt embeddings

        Args:
            prompt_embeddings (Torch tensor or numpy array): prompt embeddings table.
        """

        if not isinstance(prompt_embeddings, np.ndarray):
            raise TypeError("Only numpy array and torch tensor are allowed.")

        if self.model_dir is None:
            raise Exception("A model dir cannot be None.")

        if len(prompt_embeddings.shape) != 2:
            raise Exception("A two dimensional table for a sinlge task is only supported.")

        if torch.is_tensor(prompt_embeddings):
            prompt_embeddings = torch_to_numpy(prompt_embeddings)

        hd = self.get_hidden_size() 
        if hd is None:
            raise Exception("Model config is not available. This means that no model has been loaded.")
        else:
            if hd != prompt_embeddings.shape[1]:
                raise Exception("Model hidden size is {0} and prompt embedding table size has to match with model hidden size.".format(hd))

        np.save(os.path.join(self.model_dir, "__prompt_embeddings__.npy"), prompt_embeddings)

        self.config["builder_config"]["use_prompt_tuning"] = True
        self._save_config_file()
        self._load_prompt_table()

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
