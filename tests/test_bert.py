# Copyright (c) 2019 NVIDIA Corporation
import unittest
import os

import torch
import wget
import subprocess

from .context import nemo_nlp
from .common_setup import NeMoUnitTest

# from pytorch_transformers import BertConfig, BertModel
# from nemo_nlp.huggingface.bert import BERT

class TestBert(NeMoUnitTest):
    def test_list_pretrained_models(self):
        pretrained_models = nemo_nlp.huggingface.BERT.list_pretrained_models()
        self.assertTrue(len(pretrained_models) > 0)

        model_info = [m for m in pretrained_models if m.pretrained_model_name == 'bert-base-uncased'][0]
        # model = nemo_nlp.huggingface.BERT(model_info.parameters)

        # pt_path = 'checkpoints/bert-base-uncased.pt'
        # cfg_path = 'checkpoints/bert-base-uncased-config.json'
        # wget.download(model_info.parameters, cfg_path)
        # subprocess.run(["wget", "-r", "-nc", "-P", pt_path, model_info.location])
        # subprocess.run(["wget", "-r", "-nc", "-P", cfg_path, model_info.parameters])

        # config = BertConfig.from_json_file('checkpoints/bert-base-uncased-config.json/s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json')
        # model = BertModel(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)

    # torch.distributed.init_process_group(backend='nccl', init_method='env://')


        model = nemo_nlp.huggingface.BERT( pretrained_model_name=model_info.pretrained_model_name)

        # state_dict = torch.load(pt_path, map_location="cpu") #["model"]
        # model.load_state_dict(state_dict, strict=False)

        model.eval()
        print("Evaluation. . .")

        model = torch.jit.script(model)

        print("ONNX. . .")
        input_shape = (768,)
        shape = (1,) + input_shape
        dummy_input  = (torch.randint(0, 768, input_shape, device='cuda'),)
        onnx_name = model_info.pretrained_model_name + ".onnx"

        torch.onnx.export(model,
                          dummy_input,
                          onnx_name,
                          input_names = [],
                          output_names = [],
                          verbose=True,
                          export_params=True,
                          opset_version=9)

