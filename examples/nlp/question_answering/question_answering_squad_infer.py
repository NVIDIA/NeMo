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

To run Question Answering inference on pretrained question answering checkpoints on 1 GPU:
python question_answering_squad_infer.py
--data_file /path_to_data_dir/squad/v1.1/train-v1.1.json
--checkpoint_dir /path_to_checkpoints
--do_lower_case
"""
import argparse
import json
import os

import numpy as np

import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Squad_with_pretrained_BERT")
    parser.add_argument(
        "--data_file", type=str, help="The data file. Should be *.json",
    )
    parser.add_argument("--pretrained_model_name", type=str, help="Name of the pre-trained model")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Checkpoint directory for inference.")
    parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
    parser.add_argument(
        "--model_type", default="bert", type=str, help="model type", choices=['bert', 'roberta', 'albert']
    )
    parser.add_argument(
        "--bert_checkpoint", default=None, type=str, help="Path to BERT model checkpoint for finetuning."
    )
    parser.add_argument(
        "--tokenizer_model",
        default="tokenizer.model",
        type=str,
        help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
    )
    parser.add_argument(
        "--tokenizer",
        default="nemobert",
        type=str,
        choices=["nemobert", "sentencepiece"],
        help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--no_data_cache", action='store_true', help="When specified do not load and store cache preprocessed data.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
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
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
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
        help="File to write predictions to. Only in evaluation mode.",
    )
    args = parser.parse_args()
    return args


def create_pipeline(
    data_file,
    model,
    head,
    max_query_length,
    max_seq_length,
    doc_stride,
    batch_size,
    version_2_with_negative,
    num_gpus=1,
    batches_per_step=1,
    use_data_cache=True,
):
    data_layer = nemo_nlp.nm.data_layers.BertQuestionAnsweringDataLayer(
        mode="infer",
        version_2_with_negative=version_2_with_negative,
        batch_size=batch_size,
        tokenizer=tokenizer,
        data_file=data_file,
        max_query_length=max_query_length,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        use_cache=use_data_cache,
    )

    input_data = data_layer()

    hidden_states = model(
        input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
    )

    logits = head(hidden_states=hidden_states)

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus * batches_per_step)

    return (
        steps_per_epoch,
        [input_data.unique_ids, logits],
        data_layer,
    )


MODEL_CLASSES = {
    'bert': nemo_nlp.nm.trainables.huggingface.BERT,
    'albert': nemo_nlp.nm.trainables.huggingface.Albert,
    'roberta': nemo_nlp.nm.trainables.huggingface.Roberta,
}


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_file):
        raise FileNotFoundError("inference data not found")

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
            tokenizer = nemo_nlp.data.SentencePieceTokenizer(model_path=args.tokenizer_model)
        except Exception:
            raise ValueError(
                "Using --tokenizer=sentencepiece \
                        requires valid --tokenizer_model"
            )
        special_tokens = nemo_nlp.utils.MODEL_SPECIAL_TOKENS[args.model_type]
        tokenizer.add_special_tokens(special_tokens)
    else:
        tokenizer_cls = nemo_nlp.data.NemoBertTokenizer
        tokenizer_special_tokens = nemo_nlp.utils.MODEL_SPECIAL_TOKENS[args.model_type]
        tokenizer_name = nemo_nlp.utils.MODEL_NAMES[args.model_type]["tokenizer_name"]
        tokenizer = tokenizer_cls(
            do_lower_case=args.do_lower_case,
            pretrained_model=tokenizer_name,
            special_tokens=tokenizer_special_tokens,
            bert_derivate=args.model_type,
        )

    model_cls = MODEL_CLASSES[args.model_type]
    model_name = nemo_nlp.utils.MODEL_NAMES[args.model_type]["model_name"]

    if args.pretrained_model_name is None:
        args.pretrained_model_name = model_name

    if args.bert_config is not None:
        with open(args.bert_config) as json_file:
            config = json.load(json_file)
        model = model_cls(**config)
    else:
        """ Use this if you're using a standard BERT model.
        To see the list of pretrained models, call:
        nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()
        """
        model = model_cls(pretrained_model_name=args.pretrained_model_name)

    hidden_size = model.hidden_size

    qa_head = nemo_nlp.nm.trainables.TokenClassifier(
        hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False
    )
    if args.bert_checkpoint is not None:
        model.restore_from(args.bert_checkpoint)

    _, eval_output, eval_data_layer = create_pipeline(
        data_file=args.data_file,
        model=model,
        head=qa_head,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        version_2_with_negative=args.version_2_with_negative,
        num_gpus=args.num_gpus,
        batches_per_step=args.batches_per_step,
        use_data_cache=not args.no_data_cache,
    )
    load_from_folder = None
    if args.checkpoint_dir is not None:
        load_from_folder = args.checkpoint_dir
    evaluated_tensors = nf.infer(tensors=eval_output, checkpoint_dir=load_from_folder, cache=True)
    unique_ids = []
    logits = []
    for t in evaluated_tensors[0]:
        unique_ids.extend(t.tolist())
    for t in evaluated_tensors[1]:
        logits.extend(t.tolist())

    start_logits, end_logits = np.split(np.asarray(logits), 2, axis=-1)

    (all_predictions, all_nbest_json, scores_diff_json) = eval_data_layer.dataset.get_predictions(
        unique_ids=unique_ids,
        start_logits=start_logits,
        end_logits=end_logits,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        version_2_with_negative=args.version_2_with_negative,
        null_score_diff_threshold=args.null_score_diff_threshold,
        do_lower_case=args.do_lower_case,
    )

    if args.output_prediction_file is not None:
        with open(args.output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
