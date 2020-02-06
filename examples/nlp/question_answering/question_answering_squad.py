"""
Copyright 2018 The Google AI Language Team Authors and
The HuggingFace Inc. team.
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""

"""
Download the Squad data by running the script:
examples/nlp/scripts/get_squad.py

To finetune Squad v1.1 on pretrained BERT large uncased on 1 GPU:
python question_answering_squad.py
--train_file /path_to_data_dir/squad/v1.1/train-v1.1.json
--dev_file /path_to_data_dir/squad/v1.1/dev-v1.1.json
--work_dir /path_to_output_folder
--bert_checkpoint /path_to_bert_checkpoint
--amp_opt_level "O1"
--batch_size 24
--num_epochs 2
--lr_policy WarmupAnnealing
--lr_warmup_proportion 0.0
--optimizer adam_w
--weight_decay 0.0
--lr 3e-5
--do_lower_case

If --bert_checkpoint is not specified, training starts from
Huggingface pretrained checkpoints.

To finetune Squad v1.1 on pretrained BERT large uncased on 8 GPU:
python -m torch.distributed.launch --nproc_per_node=8 question_answering_squad.py
--amp_opt_level "O1"
--train_file /path_to_data_dir/squad/v1.1/train-v1.1.json
--dev_file /path_to_data_dir/squad/v1.1/dev-v1.1.json
--bert_checkpoint /path_to_bert_checkpoint
--batch_size 3
--num_gpus 8
--num_epochs 2
--lr_policy WarmupAnnealing
--lr_warmup_proportion 0.0
--optimizer adam_w
--weight_decay 0.0
--lr 3e-5
--do_lower_case

On Huggingface the final Exact Match (EM) and F1 scores are as follows:
Model	                EM      F1
BERT Based uncased      80.59    88.34
BERT Large uncased      83.88    90.65
"""
import argparse
import json
import os

import nemo
import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo import logging
from nemo.collections.nlp.callbacks.qa_squad_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.utils.lr_policies import get_lr_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Squad_with_pretrained_BERT")
    parser.add_argument(
        "--train_file", type=str, help="The training data file. Should be *.json",
    )
    parser.add_argument(
        "--dev_file", type=str, required=True, help="The evaluation data file. Should be *.json",
    )
    parser.add_argument(
        "--pretrained_bert_model", default="bert-base-uncased", type=str, help="Name of the pre-trained model"
    )
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Checkpoint directory for inference.")
    parser.add_argument(
        "--bert_checkpoint", default=None, type=str, help="Path to BERT model checkpoint for finetuning."
    )
    parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
    parser.add_argument(
        "--tokenizer_model",
        default="tokenizer.model",
        type=str,
        help="Path to pretrained tokenizer model," "only used if --tokenizer is sentencepiece",
    )
    parser.add_argument(
        "--tokenizer",
        default="nemobert",
        type=str,
        choices=["nemobert", "sentencepiece"],
        help="tokenizer to use, " "only relevant when using custom " "pretrained checkpoint.",
    )
    parser.add_argument("--optimizer_kind", default="adam", type=str, help="Optimizer kind")
    parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
    parser.add_argument("--lr", default=3e-5, type=float, help="The initial learning rate.")
    parser.add_argument("--lr_warmup_proportion", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Whether to lower case the input text. " "True for uncased models, False for cased models.",
    )
    parser.add_argument("--evaluation_only", action='store_true', help="Whether to only do evaluation.")
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, " "how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. "
        "Questions longer than this will be truncated to "
        "this length.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after "
        "WordPiece tokenization. Sequences longer than this "
        "will be truncated, and sequences shorter than this "
        " will be padded.",
    )
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument(
        "--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--local_rank", type=int, default=None, help="For distributed training: local_rank")
    parser.add_argument(
        "--work_dir",
        default='output_squad',
        type=str,
        help="The output directory where the " "model predictions and checkpoints " "will be written.",
    )
    parser.add_argument(
        "--save_epoch_freq",
        default=1,
        type=int,
        help="Frequency of saving checkpoint " "'-1' - epoch checkpoint won't be saved",
    )
    parser.add_argument(
        "--save_step_freq",
        default=-1,
        type=int,
        help="Frequency of saving checkpoint " "'-1' - step checkpoint won't be saved",
    )
    parser.add_argument("--loss_step_freq", default=100, type=int, help="Frequency of printing loss")
    parser.add_argument("--eval_step_freq", default=500, type=int, help="Frequency of evaluation on dev data")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that " "do not have an answer.",
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help="If null_score - best_non_null is " "greater than the threshold predict null.",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to " "generate in the nbest_predictions.json output file.",
    )
    parser.add_argument("--batches_per_step", default=1, type=int, help="Number of iterations per step.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be "
        "generated. This is needed because the start "
        "and end predictions are not conditioned "
        "on one another.",
    )
    parser.add_argument(
        "--output_prediction_file",
        type=str,
        required=False,
        default="predictions.json",
        help="File to write predictions to. " "Only in evaluation mode.",
    )
    args = parser.parse_args()
    return args


def create_pipeline(
    data_file,
    model,
    head,
    loss_fn,
    max_query_length,
    max_seq_length,
    doc_stride,
    batch_size,
    version_2_with_negative,
    num_gpus=1,
    batches_per_step=1,
    mode="train",
):
    data_layer = nemo_nlp.nm.data_layers.qa_squad_datalayer.BertQuestionAnsweringDataLayer(
        mode=mode,
        version_2_with_negative=version_2_with_negative,
        batch_size=batch_size,
        tokenizer=tokenizer,
        data_file=data_file,
        max_query_length=max_query_length,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
    )

    input_data = data_layer()

    hidden_states = model(
        input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
    )

    qa_output = head(hidden_states=hidden_states)
    loss_output = loss_fn(
        logits=qa_output, start_positions=input_data.start_positions, end_positions=input_data.end_positions
    )

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus * batches_per_step)
    return (
        loss_output.loss,
        steps_per_epoch,
        [loss_output.start_logits, loss_output.end_logits, input_data.unique_ids],
        data_layer,
    )


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.dev_file):
        raise FileNotFoundError("eval data not found. Datasets can be " "obtained using scripts/get_squad.py")
    if not args.evaluation_only and not os.path.exists(args.train_file):
        raise FileNotFoundError("train data not found. Datasets can be " "obtained using scripts/get_squad.py")

    # Instantiate neural factory with supported backend
    nf = nemo_core.NeuralModuleFactory(
        backend=nemo_core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=args.work_dir,
        create_tb_writer=True,
        files_to_copy=[__file__],
        add_time_to_log_dir=False,
    )

    if args.tokenizer == "sentencepiece":
        try:
            tokenizer = nemo_nlp.data.utilsSentencePieceTokenizer(model_path=args.tokenizer_model)
        except Exception:
            raise ValueError(
                "Using --tokenizer=sentencepiece \
                        requires valid --tokenizer_model"
            )
        tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
    elif args.tokenizer == "nemobert":
        tokenizer = nemo_nlp.data.NemoBertTokenizer(args.pretrained_bert_model)
    else:
        raise ValueError(f"received unexpected tokenizer '{args.tokenizer}'")

    if args.bert_config is not None:
        with open(args.bert_config) as json_file:
            config = json.load(json_file)
        model = nemo_nlp.nm.trainables.huggingface.BERT(**config)
    else:
        """ Use this if you're using a standard BERT model.
        To see the list of pretrained models, call:
        nemo_nlp.huggingface.BERT.list_pretrained_models()
        """
        model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_bert_model)

    hidden_size = model.hidden_size

    qa_head = nemo_nlp.nm.trainables.token_classification_nm.TokenClassifier(
        hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False
    )
    squad_loss = nemo_nlp.nm.losses.QuestionAnsweringLoss()
    if args.bert_checkpoint is not None:
        model.restore_from(args.bert_checkpoint)

    if not args.evaluation_only:
        train_loss, train_steps_per_epoch, _, _ = create_pipeline(
            data_file=args.train_file,
            model=model,
            head=qa_head,
            loss_fn=squad_loss,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            batch_size=args.batch_size,
            version_2_with_negative=args.version_2_with_negative,
            num_gpus=args.num_gpus,
            batches_per_step=args.batches_per_step,
            mode="train",
        )
        logging.info(f"training step per epoch: {train_steps_per_epoch}")
    _, _, eval_output, eval_data_layer = create_pipeline(
        data_file=args.dev_file,
        model=model,
        head=qa_head,
        loss_fn=squad_loss,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        version_2_with_negative=args.version_2_with_negative,
        num_gpus=args.num_gpus,
        batches_per_step=args.batches_per_step,
        mode="dev",
    )

    if not args.evaluation_only:
        logging.info(f"steps_per_epoch = {train_steps_per_epoch}")
        callback_train = nemo_core.SimpleLossLoggerCallback(
            tensors=[train_loss],
            print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=args.loss_step_freq,
            tb_writer=nf.tb_writer,
        )

        ckpt_callback = nemo_core.CheckpointCallback(
            folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
        )
        callbacks_eval = nemo_core.EvaluatorCallback(
            eval_tensors=eval_output,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x,
                eval_data_layer=eval_data_layer,
                do_lower_case=args.do_lower_case,
                n_best_size=args.n_best_size,
                max_answer_length=args.max_answer_length,
                version_2_with_negative=args.version_2_with_negative,
                null_score_diff_threshold=args.null_score_diff_threshold,
            ),
            tb_writer=nf.tb_writer,
            eval_step=args.eval_step_freq,
        )

        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.num_epochs * train_steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
        )

        nf.train(
            tensors_to_optimize=[train_loss],
            callbacks=[callback_train, ckpt_callback, callbacks_eval],
            lr_policy=lr_policy_fn,
            optimizer=args.optimizer_kind,
            batches_per_step=args.batches_per_step,
            optimization_params={"num_epochs": args.num_epochs, "lr": args.lr},
        )
    else:

        if args.checkpoint_dir is not None:
            load_from_folder = args.checkpoint_dir
        evaluated_tensors = nf.infer(tensors=eval_output, checkpoint_dir=load_from_folder, cache=True)
        unique_ids = []
        start_logits = []
        end_logits = []
        for t in evaluated_tensors[2]:
            unique_ids.extend(t.tolist())
        for t in evaluated_tensors[0]:
            start_logits.extend(t.tolist())
        for t in evaluated_tensors[1]:
            end_logits.extend(t.tolist())

        exact_match, f1, all_predictions = eval_data_layer.dataset.evaluate(
            unique_ids=unique_ids,
            start_logits=start_logits,
            end_logits=end_logits,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            version_2_with_negative=args.version_2_with_negative,
            null_score_diff_threshold=args.null_score_diff_threshold,
            do_lower_case=args.do_lower_case,
        )
        logging.info(f"exact_match: {exact_match}, f1: {f1}")
        if args.output_prediction_file is not None:
            with open(args.output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
