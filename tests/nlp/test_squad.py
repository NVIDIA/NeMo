# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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
# =============================================================================

import json
import os
import shutil

from examples.nlp.scripts.get_squad import SquadDownloader

import nemo
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.nm.data_layers.qa_squad_datalayer
import nemo.collections.nlp.nm.trainables.common.token_classification_nm
from nemo.collections.nlp.callbacks.qa_squad_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.utils.lr_policies import get_lr_policy
from tests.common_setup import NeMoUnitTest

print(dir(nemo_nlp))


class TestSquad(NeMoUnitTest):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/nlp'))
        if not os.path.exists(data_folder):
            print(f"mkdir {data_folder}")
            os.mkdir(data_folder)

        squad_folder = data_folder + '/squad'
        if not os.path.exists(squad_folder):
            print("Extracting Squad data to: {0}".format(squad_folder))
            squad_dl = SquadDownloader(data_folder)
            squad_dl.download()

            squad_v1_dev_file = os.path.join(squad_folder, 'v1.1/dev-v1.1.json')
            squad_v1_train_file = os.path.join(squad_folder, 'v1.1/train-v1.1.json')
            squad_v2_dev_file = os.path.join(squad_folder, 'v2.0/dev-v2.0.json')
            squad_v2_train_file = os.path.join(squad_folder, 'v2.0/train-v2.0.json')
            with open(squad_v1_dev_file, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            data["data"] = [data["data"][0]]
            with open(squad_v1_dev_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file)
            with open(squad_v1_train_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file)

            with open(squad_v2_dev_file, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            data["data"] = [data["data"][0]]
            with open(squad_v2_dev_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file)
            with open(squad_v2_train_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        squad_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/nlp/squad'))
        if os.path.exists(squad_folder):
            shutil.rmtree(squad_folder)

    def test_squad_v1(self):
        version_2_with_negative = False
        pretrained_bert_model = 'bert-base-uncased'
        batch_size = 3
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/nlp/squad/v1.1'))
        max_query_length = 64
        max_seq_length = 384
        doc_stride = 128
        max_steps = 100
        lr_warmup_proportion = 0
        eval_step_freq = 50
        lr = 3e-6
        do_lower_case = True
        n_best_size = 5
        max_answer_length = 20
        null_score_diff_threshold = 0.0

        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_bert_model)
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False
        )
        model = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name=pretrained_bert_model)
        hidden_size = model.local_parameters["hidden_size"]
        qa_head = nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier(
            hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False
        )
        squad_loss = nemo_nlp.nm.losses.QuestionAnsweringLoss()

        data_layer = nemo.collections.nlp.nm.data_layers.qa_squad_datalayer.BertQuestionAnsweringDataLayer(
            mode='train',
            version_2_with_negative=version_2_with_negative,
            batch_size=batch_size,
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )

        (input_ids, input_type_ids, input_mask, start_positions, end_positions, _) = data_layer()

        hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        qa_output = qa_head(hidden_states=hidden_states)
        loss, _, _ = squad_loss(logits=qa_output, start_positions=start_positions, end_positions=end_positions)

        data_layer_eval = nemo.collections.nlp.nm.data_layers.qa_squad_datalayer.BertQuestionAnsweringDataLayer(
            mode='dev',
            version_2_with_negative=version_2_with_negative,
            batch_size=batch_size,
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )
        (
            input_ids_eval,
            input_type_ids_eval,
            input_mask_eval,
            start_positions_eval,
            end_positions_eval,
            unique_ids_eval,
        ) = data_layer_eval()

        hidden_states_eval = model(
            input_ids=input_ids_eval, token_type_ids=input_type_ids_eval, attention_mask=input_mask_eval
        )

        qa_output_eval = qa_head(hidden_states=hidden_states_eval)
        _, start_logits_eval, end_logits_eval = squad_loss(
            logits=qa_output_eval, start_positions=start_positions_eval, end_positions=end_positions_eval
        )
        eval_output = [start_logits_eval, end_logits_eval, unique_ids_eval]

        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss],
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=10,
            tb_writer=neural_factory.tb_writer,
        )

        callbacks_eval = nemo.core.EvaluatorCallback(
            eval_tensors=eval_output,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x,
                eval_data_layer=data_layer_eval,
                do_lower_case=do_lower_case,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                version_2_with_negative=version_2_with_negative,
                null_score_diff_threshold=null_score_diff_threshold,
            ),
            tb_writer=neural_factory.tb_writer,
            eval_step=eval_step_freq,
        )

        lr_policy_fn = get_lr_policy('WarmupAnnealing', total_steps=max_steps, warmup_ratio=lr_warmup_proportion)

        neural_factory.train(
            tensors_to_optimize=[loss],
            callbacks=[callback_train, callbacks_eval],
            lr_policy=lr_policy_fn,
            optimizer='adam_w',
            optimization_params={"max_steps": max_steps, "lr": lr},
        )

    def test_squad_v2(self):
        version_2_with_negative = True
        pretrained_bert_model = 'bert-base-uncased'
        batch_size = 3
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/nlp/squad/v2.0'))
        max_query_length = 64
        max_seq_length = 384
        doc_stride = 128
        max_steps = 100
        lr_warmup_proportion = 0
        eval_step_freq = 50
        lr = 3e-6
        do_lower_case = True
        n_best_size = 5
        max_answer_length = 20
        null_score_diff_threshold = 0.0

        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_bert_model)
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False
        )
        model = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name=pretrained_bert_model)
        hidden_size = model.local_parameters["hidden_size"]
        qa_head = nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier(
            hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False
        )
        squad_loss = nemo_nlp.nm.losses.QuestionAnsweringLoss()

        data_layer = nemo.collections.nlp.nm.data_layers.qa_squad_datalayer.BertQuestionAnsweringDataLayer(
            mode='train',
            version_2_with_negative=version_2_with_negative,
            batch_size=batch_size,
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )

        (input_ids, input_type_ids, input_mask, start_positions, end_positions, _) = data_layer()

        hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        qa_output = qa_head(hidden_states=hidden_states)
        loss, _, _ = squad_loss(logits=qa_output, start_positions=start_positions, end_positions=end_positions)

        data_layer_eval = nemo.collections.nlp.nm.data_layers.qa_squad_datalayer.BertQuestionAnsweringDataLayer(
            mode='dev',
            version_2_with_negative=version_2_with_negative,
            batch_size=batch_size,
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )
        (
            input_ids_eval,
            input_type_ids_eval,
            input_mask_eval,
            start_positions_eval,
            end_positions_eval,
            unique_ids_eval,
        ) = data_layer_eval()

        hidden_states_eval = model(
            input_ids=input_ids_eval, token_type_ids=input_type_ids_eval, attention_mask=input_mask_eval
        )

        qa_output_eval = qa_head(hidden_states=hidden_states_eval)
        _, start_logits_eval, end_logits_eval = squad_loss(
            logits=qa_output_eval, start_positions=start_positions_eval, end_positions=end_positions_eval
        )
        eval_output = [start_logits_eval, end_logits_eval, unique_ids_eval]

        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss],
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=10,
            tb_writer=neural_factory.tb_writer,
        )

        callbacks_eval = nemo.core.EvaluatorCallback(
            eval_tensors=eval_output,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x,
                eval_data_layer=data_layer_eval,
                do_lower_case=do_lower_case,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                version_2_with_negative=version_2_with_negative,
                null_score_diff_threshold=null_score_diff_threshold,
            ),
            tb_writer=neural_factory.tb_writer,
            eval_step=eval_step_freq,
        )

        lr_policy_fn = get_lr_policy('WarmupAnnealing', total_steps=max_steps, warmup_ratio=lr_warmup_proportion)

        neural_factory.train(
            tensors_to_optimize=[loss],
            callbacks=[callback_train, callbacks_eval],
            lr_policy=lr_policy_fn,
            optimizer='adam_w',
            optimization_params={"max_steps": max_steps, "lr": lr},
        )
