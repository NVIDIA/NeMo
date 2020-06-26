# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

'''
LaserTagger model training and inference main file
'''

import sys

import torch
from official_lasertagger import bert_example, score_lib, tagging, tagging_converter, utils
from rouge_score import rouge_scorer, scoring

import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.nlp.callbacks.lasertagger_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.nm.data_layers.lasertagger_datalayer import LaserTaggerDataLayer
from nemo.core import EvaluatorCallback, NeuralModuleFactory, SimpleLossLoggerCallback, WeightShareTransform
from nemo.core.callbacks import CheckpointCallback
from nemo.utils import NemoArgParser
from nemo.utils.lr_policies import PolynomialDecayAnnealing

sys.modules['bert_example'] = bert_example
sys.modules['score_lib'] = score_lib
sys.modules['tagging'] = tagging
sys.modules['tagging_converter'] = tagging_converter
sys.modules['utils'] = utils


def parse_args():
    '''
    LaserTagger model trainer and inference argument parser
    '''
    parser = NemoArgParser(description='LaserTagger')
    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    # LaserTagger model config args
    parser.add_argument(
        "--use_t2t_decoder",
        type=bool,
        help="Use AutoRegressive decoder instead of feed-forward decoder for decoding",
        default=False,
    )
    parser.add_argument("--decoder_hidden_size", type=float, help="Dimensionality of the decoder layers", default=768)
    parser.add_argument(
        "--decoder_filter_size",
        type=int,
        help="Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer Decoder",
        default=3072,
    )
    parser.add_argument(
        "--decoder_num_hidden_layers", type=int, help="Number of hidden layers in the Transformer Decoder", default=1,
    )
    parser.add_argument(
        "--decoder_num_attention_heads",
        type=int,
        help="Number of attention heads for each attention layer in the Transformer Decoder",
        default=4,
    )
    parser.add_argument(
        "--use_full_attention",
        type=bool,
        help="Use a full attention over the sequence of encoder activations",
        default=False,
    )
    parser.add_argument(
        "--decoder_ffn_dropout", type=float, help="Decoder Feed-forward layer dropout prob", default=0.1
    )
    parser.add_argument(
        "--decoder_attn_score_dropout", type=float, help="Decoder Self-Attention layer dropout prob", default=0.1
    )
    parser.add_argument(
        "--decoder_embedding_dropout", type=float, help="Decoder Embedding layer dropout prob", default=0.1
    )
    parser.add_argument(
        "--decoder_hidden_act", type=str, help="Decoder Position-Wise Feed-forward layer activation", default='relu'
    )

    # Training arguments
    parser_train = subparsers.add_parser('train', help='train a lasertagger model')
    parser_train.set_defaults(
        optimizer="adam_w",
        amp_opt_level="O1",
        num_epochs=3,
        batch_size=8,
        eval_batch_size=8,
        lr=3e-5,
        weight_decay=0.1,
        iter_per_step=1,
        warmup_proportion=0.1,
        checkpoint_save_freq=1000,
        eval_freq=1000,
        work_dir='outputs/msr_ab_sum/lt',
    )
    parser_train.add_argument(
        "--train_file_preprocessed", type=str, help="The path to training pkl file",
    )
    parser_train.add_argument(
        "--eval_file_preprocessed", type=str, help="The path to evaluation pkl file",
    )
    parser_train.add_argument(
        "--test_file_preprocessed", type=str, help="The path to test pkl file",
    )
    parser_train.add_argument(
        "--label_map_file",
        type=str,
        help="Path to the label map file. Either a JSON file ending with '.json', \
				that maps each possible tag to an ID, or a text file that \
				has one tag per line.",
    )
    parser_train.add_argument(
        '--pretrained_model_name',
        default='bert-base-cased',
        type=str,
        help='Name of the pre-trained model',
        choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
    )
    parser_train.add_argument("--max_seq_length", default=128, type=int)
    parser_train.add_argument("--warmup_steps", default=4500, type=int)

    # Inference arguments
    parser_infer = subparsers.add_parser('infer', help='infer on a dataset using lasertagger saved model checkpoint')
    parser_infer.set_defaults(
        amp_opt_level="O1", batch_size=64, beam_size=4, length_penalty=0.0, work_dir='outputs/msr_ab_sum/lt'
    )
    parser_infer.add_argument(
        "--test_file", type=str, help="The path to test tsv file",
    )
    parser_infer.add_argument(
        "--test_file_preprocessed", type=str, help="The path to preprocessed test pkl file",
    )
    parser_infer.add_argument(
        "--label_map_file",
        type=str,
        help="Path to the label map file. Either a JSON file ending with '.json', \
				that maps each possible tag to an ID, or a text file that \
				has one tag per line.",
    )
    parser_infer.add_argument(
        '--pretrained_model_name',
        default='bert-base-cased',
        type=str,
        help='Name of the pre-trained model',
        choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
    )
    parser_infer.add_argument("--max_seq_length", default=128, type=int)

    return parser.parse_args()


def _calculate_steps(num_examples, batch_size, num_epochs, warmup_proportion=0):
    """Calculates the number of steps.

  Args:
	num_examples: Number of examples in the dataset.
	batch_size: Batch size.
	num_epochs: How many times we should go through the dataset.
	warmup_proportion: Proportion of warmup steps.

  Returns:
	Tuple (number of steps, number of warmup steps).
  """
    steps = int(num_examples / batch_size * num_epochs)
    warmup_steps = int(warmup_proportion * steps)
    return steps, warmup_steps


if __name__ == "__main__":

    args = parse_args()

    if args.command == 'train':
        train_examples = torch.load(args.train_file_preprocessed)
        if args.eval_file_preprocessed:
            eval_examples, eval_special_tokens = torch.load(args.eval_file_preprocessed)
    test_examples, test_special_tokens = torch.load(args.test_file_preprocessed)

    label_map = utils.read_label_map(args.label_map_file)
    num_tags = len(label_map)

    nf = NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=args.work_dir,
        create_tb_writer=True,
        files_to_copy=[__file__],
        add_time_to_log_dir=False,
    )

    encoder = nemo_nlp.nm.trainables.huggingface.get_huggingface_lm_model(
        pretrained_model_name=args.pretrained_model_name
    )

    tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)

    hidden_size = encoder.hidden_size

    # Size of the output vocabulary which contains the tags + begin and end
    # tokens used by the Transformer decoder.
    output_vocab_size = num_tags

    decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
        d_model=args.decoder_hidden_size,
        d_inner=args.decoder_filter_size,
        num_layers=args.decoder_num_hidden_layers,
        num_attn_heads=args.decoder_num_attention_heads,
        ffn_dropout=args.decoder_ffn_dropout,
        vocab_size=output_vocab_size,
        attn_score_dropout=args.decoder_attn_score_dropout,
        max_seq_length=args.max_seq_length,
        embedding_dropout=args.decoder_embedding_dropout,
        hidden_act=args.decoder_hidden_act,
        use_full_attention=args.use_full_attention,
    )

    logits = nemo_nlp.nm.trainables.TokenClassifier(
        hidden_size, num_classes=output_vocab_size, num_layers=1, log_softmax=False, dropout=0.1
    )

    loss_fn = CrossEntropyLossNM(logits_ndim=3)
    loss_eval_metric = CrossEntropyLossNM(logits_ndim=3, reduction='none')

    if args.command == "infer":
        beam_search = nemo_nlp.nm.trainables.BeamSearchTranslatorNM(
            decoder=decoder,
            log_softmax=logits,
            max_seq_length=args.max_seq_length,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            bos_token=tokenizer.bos_id,
            pad_token=tokenizer.pad_id,
            eos_token=tokenizer.eos_id,
        )

    # tie all embeddings weights
    if args.use_t2t_decoder:
        decoder.tie_weights_with(
            encoder,
            weight_names=["embedding_layer.token_embedding.weight"],
            name2name_and_transform={
                "embedding_layer.token_embedding.weight": (
                    "bert.embeddings.word_embeddings.weight",
                    WeightShareTransform.SAME,
                )
            },
        )
        decoder.tie_weights_with(
            encoder,
            weight_names=["embedding_layer.position_embedding.weight"],
            name2name_and_transform={
                "embedding_layer.position_embedding.weight": (
                    "bert.embeddings.position_embeddings.weight",
                    WeightShareTransform.SAME,
                )
            },
        )

    def create_pipeline(dataset, batch_size, mode):
        data_layer = LaserTaggerDataLayer(dataset, args.use_t2t_decoder, batch_size, shuffle=(mode == "train"))
        (input_ids, input_mask, segment_ids, tgt_ids, labels_mask, labels, input_mask_tgt) = data_layer()
        src_hiddens = encoder(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        tgt_hiddens = decoder(
            input_ids_tgt=tgt_ids,
            hidden_states_src=src_hiddens,
            input_mask_src=input_mask,
            input_mask_tgt=input_mask_tgt,
        )
        log_softmax = logits(hidden_states=tgt_hiddens) if args.use_t2t_decoder else logits(hidden_states=src_hiddens)
        if mode != "infer":
            loss = loss_fn(logits=log_softmax, labels=labels, loss_mask=labels_mask)
            per_example_loss = loss_eval_metric(logits=log_softmax, labels=labels, loss_mask=labels_mask)
            return [loss, per_example_loss, log_softmax, labels, labels_mask]
        else:
            if args.use_t2t_decoder:
                return [beam_search(hidden_states_src=src_hiddens, input_mask_src=input_mask)]
            else:
                return [log_softmax]

    if args.command == "train":
        # training pipeline
        train_tensors = create_pipeline(train_examples, args.batch_size, mode="train")

        # evaluation pipelines
        eval_tensors = create_pipeline(eval_examples, args.eval_batch_size, mode="eval")

        def print_loss(x):
            loss = x[0].item()
            logging.info("Training loss: {:.4f}".format(loss))

        # callbacks
        callback_train = SimpleLossLoggerCallback(
            tensors=[train_tensors[0]],
            step_freq=100,
            print_func=print_loss,
            get_tb_values=lambda x: [["loss", x[0]]],
            tb_writer=nf.tb_writer,
        )

        callbacks = [callback_train]

        # for eval_examples in args.eval_file_preprocessed:
        callback_eval = EvaluatorCallback(
            eval_tensors=eval_tensors,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=args.eval_freq,
            tb_writer=nf.tb_writer,
        )
        if eval_examples:
            callbacks.append(callback_eval)

        checkpointer_callback = CheckpointCallback(folder=args.work_dir, step_freq=args.checkpoint_save_freq)
        callbacks.append(checkpointer_callback)

        max_steps, warmup_steps = _calculate_steps(
            len(train_examples), args.batch_size, args.num_epochs, args.warmup_proportion
        )

        # define learning rate decay policy
        lr_policy = PolynomialDecayAnnealing(total_steps=max_steps, min_lr=0, warmup_steps=warmup_steps)

        # Create trainer and execute training action
        nf.train(
            tensors_to_optimize=[train_tensors[0]],
            callbacks=callbacks,
            optimizer=args.optimizer,
            lr_policy=lr_policy,
            optimization_params={
                "num_epochs": args.num_epochs,
                "max_steps": max_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            batches_per_step=args.iter_per_step,
        )

    elif args.command == 'infer':
        tensors_pred = create_pipeline(test_examples, args.batch_size, mode="infer")
        computed_tensors = nf.infer(tensors=tensors_pred, checkpoint_dir=args.work_dir)

        id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}

        results = []
        for i in computed_tensors[0]:
            if args.use_t2t_decoder:
                results.extend((i[:, 1:]).cpu().numpy().tolist())
            else:
                results.extend(torch.argmax(i, dim=-1).int().cpu().numpy().tolist())

        # compute and realize predictions with LaserTagger
        sources, predictions, target_lists = [], [], []
        logging.info("Saving predictions to " + args.work_dir + "/pred.txt")
        with open(args.work_dir + "/pred.txt", 'w') as f:
            for i, example in enumerate(test_examples):
                example.features['labels'] = results[i]
                example.features['labels_mask'] = [0] + [1] * (len(results[i]) - 2) + [0]
                labels = [id_2_tag[label_id] for label_id in example.get_token_labels()]
                prediction = example.editing_task.realize_output(labels)
                predictions.append(prediction)
                f.write(prediction + "\n")

        with open(args.test_file, 'r') as f:
            for line in f:
                source, *targets = line.rstrip('\n').split('\t')
                ## TODO: For bert uncased models
                # if lowercase:
                # 	source = source.lower()
                # 	pred = pred.lower()
                # 	target = t.lower()
                sources.append(source)
                target_lists.append(targets)

        # Exact and SARI scores
        exact = score_lib.compute_exact_score(predictions, target_lists)
        sari, keep, addition, deletion = score_lib.compute_sari_scores(sources, predictions, target_lists)
        print(f'Exact score:     {100*exact:.3f}')
        print(f'SARI score:      {100*sari:.3f}')
        print(f' KEEP score:     {100*keep:.3f}')
        print(f' ADDITION score: {100*addition:.3f}')
        print(f' DELETION score: {100*deletion:.3f}')

        # ROUGE-L scores
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        scores = []
        for target, pred in zip(target_lists, predictions):
            aggregator.add_scores(scorer.score(target[0], pred))

        aggregates = aggregator.aggregate()

        print("\nROUGE scores:")
        print("----------------------------------------------------------------")
        print("score_type\t\tlow\t\tmid\t\thigh")
        print("----------------------------------------------------------------")
        for score_type, aggregate in sorted(aggregates.items()):
            print(
                "%s-Recall:   \t%f\t%f\t%f"
                % (score_type, aggregate.low.recall, aggregate.mid.recall, aggregate.high.recall)
            )
            print(
                "%s-Precision:\t%f\t%f\t%f"
                % (score_type, aggregate.low.precision, aggregate.mid.precision, aggregate.high.precision)
            )
            print(
                "%s-F_measure:\t%f\t%f\t%f"
                % (score_type, aggregate.low.fmeasure, aggregate.mid.fmeasure, aggregate.high.fmeasure)
            )
