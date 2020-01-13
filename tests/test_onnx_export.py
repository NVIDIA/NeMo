# Copyright (c) 2019 NVIDIA Corporation
import unittest
import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import *

from .context import nemo
from .common_setup import NeMoUnitTest

import os
import sys
import subprocess
import runpy
import wget
import timeit
import numpy as np

# git clone git@github.com:microsoft/onnxruntime.git
# cd onnxruntime
# ./build.sh --update --build --config RelWithDebInfo --build_shared_lib --parallel --use_cuda \
#            --cudnn_home /usr/lib/x86_64-linux-gnu --cuda_home /usr/local/cuda --enable_pybind --build_wheel
# pip install --upgrade ./build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-1.1.0-cp37-cp37m-linux_x86_64.whl
import onnxruntime as ort

import nemo
from nemo.core.neural_factory import Optimization
from nemo.backends.pytorch.common import CrossEntropyLoss, MSELoss
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp import GlueDataLayerClassification, GlueDataLayerRegression
from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
from nemo_nlp.utils.callbacks.glue import \
    eval_iter_callback, eval_epochs_done_callback

from nemo_nlp.data.datasets.utils import processors, output_modes

class TestONNXExport(NeMoUnitTest):
    
    def __init__(self, name):
        super().__init__(name)
        self.data_dir = './MRPC_data'
        self.task_name = 'mrpc'
        self.work_dir = 'output_glue'
        self.local_rank = None
        self.amp_opt_level = Optimization.mxprO0
        self.pretrained_bert_model = 'bert-base-cased'
        self.bert_checkpoint = './checkpoints/BERT-EPOCH-2.pt'
        self.onnx_checkpoint = '../bert-base-cased.onnx'
        self.classifier_checkpoint = './checkpoints/SequenceClassifier-EPOCH-2.pt'
        self.num_gpus = 1
        self.dataset_type = 'GLUEDataset'
        self.batch_size = 8
        self.max_seq_length = 128
        self.loss_step_freq = 25
        self.save_epoch_freq = 1
        self.save_step_freq = -1
        self.lr_policy = 'WarmupAnnealing'
        self.num_epochs = 1
        self.lr_warmup_proportion = 0.1
        self.optimizer_kind = 'adam'
        self.lr = 5.e-5

    def create_pipeline(self, evaluate=False, processor_id=0):
        processor = self.task_processors[processor_id]
        data_layer = 'GlueDataLayerClassification'
        if self.output_mode == 'regression':
            data_layer = 'GlueDataLayerRegression'

        data_layer = getattr(sys.modules[__name__], data_layer)

        data_layer = data_layer(
            dataset_type=self.dataset_type,
            processor=processor,
            evaluate=evaluate,
            batch_size=self.batch_size,
            num_workers=0,
            local_rank=self.local_rank,
            tokenizer=self.tokenizer,
            data_dir=os.path.join(self.data_dir, 'glue_data', self.task_name.upper()),
            max_seq_length=self.max_seq_length,
            token_params=self.token_params)

        input_ids, input_type_ids, input_mask, labels = data_layer()

        hidden_states = self.model(input_ids=input_ids,
                              token_type_ids=input_type_ids,
                              attention_mask=input_mask)

        """
        For STS-B (regressiont tast), the pooler_output represents a is single
        number prediction for each sequence.
        The rest of GLUE tasts are classification tasks; the pooler_output
        represents logits.
        """
        pooler_output = self.pooler(hidden_states=hidden_states)
        if self.task_name == 'sts-b':
            loss = self.glue_loss(preds=pooler_output, labels=labels)
        else:
            loss = self.glue_loss(logits=pooler_output, labels=labels)

        steps_per_epoch = len(data_layer) // (self.batch_size * self.num_gpus)
        return loss, steps_per_epoch, data_layer, [pooler_output, labels]

    def setUp(self):
        super().setUp()

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            wget.download("https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw"
                          "/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py", self.data_dir)
            curr_dir = os.getcwd()
            os.chdir(self.data_dir)
            subprocess.call("python ./download_glue_data.py --data_dir glue_data --tasks " + self.task_name.upper(),
                            shell=True)
            os.chdir(curr_dir)

# FIXME
# Here comes the code downloading checkpoints/BERT-EPOCH-2.pt and checkpoints/SequenceClassifier-EPOCH-2.pt

        self.work_dir = os.path.join(self.work_dir, self.task_name.upper())

        """
        Prepare GLUE task
        MNLI task has two separate dev sets: matched and mismatched
        """
        if self.task_name == 'mnli':
            eval_task_names = ("mnli", "mnli-mm")
            self.task_processors = (processors["mnli"](), processors["mnli-mm"]())
        else:
            eval_task_names = (self.task_name,)
            self.task_processors = (processors[self.task_name](),)

        label_list = self.task_processors[0].get_labels()
        self.num_labels = len(label_list)
        self.output_mode = output_modes[self.task_name]

        # Instantiate neural factory with supported backend
        nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                           local_rank=self.local_rank,
                                           optimization_level=self.amp_opt_level,
                                           log_dir=self.work_dir,
                                           create_tb_writer=True,
                                           files_to_copy=[__file__],
                                           add_time_to_log_dir=True)

        self.tokenizer = NemoBertTokenizer(self.pretrained_bert_model)
        self.model = nemo_nlp.huggingface.BERT(pretrained_model_name=self.pretrained_bert_model)

        self.hidden_size = self.model.local_parameters["hidden_size"]

        # uses [CLS] token for classification (the first token)
        if self.task_name == 'sts-b':
            self.pooler = nemo_nlp.SequenceRegression(hidden_size=self.hidden_size)
            self.glue_loss = MSELoss()
        else:
            self.pooler = nemo_nlp.SequenceClassifier(hidden_size=self.hidden_size,
                                                 num_classes=self.num_labels,
                                                 log_softmax=False)
            self.glue_loss = CrossEntropyLoss()

        if self.bert_checkpoint and self.classifier_checkpoint:
            nf.logger.info(f"restoring model from {self.bert_checkpoint} and {self.classifier_checkpoint}")
            self.model.restore_from(self.bert_checkpoint)
            self.pooler.restore_from(self.classifier_checkpoint)

        self.token_params = {'bos_token': None,
                        'eos_token': '[SEP]',
                        'pad_token': '[PAD]',
                        'cls_token': '[CLS]'}
    
        self.train_loss, self.steps_per_epoch, self.train_data_layer, _ = self.create_pipeline()
        _, _, eval_data_layer, eval_tensors = self.create_pipeline(evaluate=True)
    
        callbacks_eval = [nemo.core.EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x:
            eval_epochs_done_callback(x, self.work_dir, eval_task_names[0]),
            tb_writer=nf.tb_writer,
            eval_step=self.steps_per_epoch)]
    
        """
        MNLI task has two dev sets: matched and mismatched
        Create additional callback and data layer for MNLI mismatched dev set
        """
        if self.task_name == 'mnli':
            _, _, eval_data_layer_mm, eval_tensors_mm = self.create_pipeline(
                evaluate = True,
                processor_id = 1)
            callbacks_eval.append(nemo.core.EvaluatorCallback(
                eval_tensors=eval_tensors_mm,
                user_iter_callback=lambda x, y: eval_iter_callback(x, y),
                user_epochs_done_callback=lambda x:
                eval_epochs_done_callback(x, self.work_dir, eval_task_names[1]),
                tb_writer=nf.tb_writer,
                eval_step=self.steps_per_epoch))
    
        nf.logger.info(f"steps_per_epoch = {self.steps_per_epoch}")
        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[self.train_loss],
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=self.loss_step_freq,
            tb_writer=nf.tb_writer)

        lr_policy_fn = get_lr_policy(self.lr_policy,
                                     total_steps=self.num_epochs * self.steps_per_epoch,
                                     warmup_ratio=self.lr_warmup_proportion)

        nf.train(tensors_to_optimize=[self.train_loss],
                 callbacks=[callback_train],
                 lr_policy=lr_policy_fn,
                 optimizer=self.optimizer_kind,
                 optimization_params={"num_epochs": self.num_epochs,
                                      "lr": self.lr})
        self.model.eval()

    def test_onnx_export(self):
        input_ids, input_type_ids, input_mask, labels = self.train_data_layer.dataset.__getitem__(0)
        input_ids = np.reshape(input_ids, (1, input_ids.shape[0]))
        input_type_ids = np.reshape(input_type_ids, (1, input_type_ids.shape[0]))
        input_mask = np.reshape(input_mask, (1, input_mask.shape[0])).astype(int)
        dummy_input = (torch.randint(low=0, high=16, size=input_ids.shape).cuda(),
                       torch.randint(low=0, high=1, size=input_type_ids.shape).cuda(),
                       torch.randint(low=0, high=1, size=input_mask.shape).cuda())
        dummy_output = (torch.randn(size=(1, 128, 768)).cuda(),)

        #Regular forward
        hidden_states = self.model.forward(input_ids=torch.from_numpy(input_ids).cuda(),
                                           token_type_ids=torch.from_numpy(input_type_ids).cuda(),
                                           attention_mask=torch.from_numpy(input_mask).cuda())
        # ORT inference
        onnx_name = "glueBERT.onnx"
        torch.onnx.export(self.model,
                          dummy_input,
                          onnx_name,
                          input_names=['inputs',
                                       'token_type_ids',
                                       'attention_mask'],
                          output_names=['hidden_states'],
                          verbose=False,
                          export_params=True,
                          example_outputs=dummy_output,
                          opset_version=10)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        ort_session = ort.InferenceSession(onnx_name, sess_options)

        hidden_states_o = ort_session.run(None, {'inputs': input_ids,
                                                 'token_type_ids': input_type_ids,
                                                 'attention_mask': input_mask})
        hidden_states_o = torch.from_numpy(hidden_states_o[0]).cuda()

        self.assertLess((hidden_states_o - hidden_states).norm(p=2), 5.e-4)
