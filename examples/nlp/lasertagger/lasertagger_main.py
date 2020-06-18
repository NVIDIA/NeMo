import json
import math
import sys

import torch
from examples.nlp.lasertagger.official_lasertagger import bert_example, score_lib, tagging, tagging_converter, utils

import nemo
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.tokenizers.tokenizer_utils
import nemo.core as nemo_core
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.nlp.callbacks.lasertagger_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.nm.data_layers.lasertagger_datalayer import LaserTaggerDataLayer
from nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm import BERT
from nemo.core import WeightShareTransform
from nemo.core.callbacks import CheckpointCallback
from nemo.utils.lr_policies import PolynomialDecayAnnealing, get_lr_policy

sys.modules['bert_example'] = bert_example
sys.modules['score_lib'] = score_lib
sys.modules['tagging'] = tagging
sys.modules['tagging_converter'] = tagging_converter
sys.modules['utils'] = utils


def parse_args():
    parser = nemo.utils.NemoArgParser(description='LaserTagger')
    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    parser_train = subparsers.add_parser('train', help='train a lasertagger model')
    parser_train.set_defaults(
        optimizer="adam_w",
        amp_opt_level="O1",
        num_epochs=3,
        batch_size=64,
        eval_batch_size=8,
        lr=3e-5,
        weight_decay=0.1,
        iter_per_step=1,
        checkpoint_save_freq=1000,
        work_dir='outputs/lt-2',
        eval_freq=1000,
        pretrained_model_name='bert-base-cased',
    )
    parser_train.add_argument(
        "--train_file", type=str, help="The path to training pkl file",
    )
    parser_train.add_argument(
        "--eval_file", type=str, help="The path to evaluation pkl file",
    )
    parser_train.add_argument(
        "--test_file", type=str, help="The path to test pkl file",
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
    parser_train.add_argument(
        '--bert_checkpoint', default='bert-base-cased', type=str, help='Path to the pre-trained model checkpoint'
    )
    parser_train.add_argument(
        "--model_config_file", default=None, type=str, help="Path to LaserTagger config file in json format"
    )
    parser_train.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
    parser_train.add_argument("--max_seq_length", default=128, type=int)
    parser_train.add_argument("--warmup_steps", default=4500, type=int)

    parser_infer = subparsers.add_parser('infer', help='infer on a dataset using lasertagger saved model checkpoint')
    parser_infer.set_defaults(
        amp_opt_level="O1", batch_size=64, work_dir='outputs/lt-2', pretrained_model_name='bert-base-cased'
    )
    parser_infer.add_argument(
        "--test_file_raw", type=str, help="The path to test tsv file",
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
    parser_infer.add_argument(
        '--bert_checkpoint', default='bert-base-cased', type=str, help='Path to the pre-trained model checkpoint'
    )
    parser_infer.add_argument(
        "--model_config_file", default=None, type=str, help="Path to LaserTagger config file in json format"
    )
    parser_infer.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
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
        train_examples, num_train_examples = torch.load(args.train_file)
        if args.eval_file:
            eval_examples, num_eval_examples, eval_special_tokens = torch.load(args.eval_file)
    test_examples, num_test_examples, test_special_tokens = torch.load(args.test_file_preprocessed)

    label_map = utils.read_label_map(args.label_map_file)
    num_tags = len(label_map)

    config = None
    with open(args.model_config_file, "r", encoding="utf-8") as reader:
        text = reader.read()
        config = json.loads(text)

    nf = nemo_core.NeuralModuleFactory(
        backend=nemo_core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=args.work_dir,
        create_tb_writer=True,
        files_to_copy=[__file__],
        add_time_to_log_dir=False,
    )

    encoder = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_model_name)

    tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": test_special_tokens})
    # encoder.resize_token_embeddings(len(tokenizer))

    vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
    tokens_to_add = vocab_size - tokenizer.vocab_size

    encoder = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_model_name)

    device = encoder.bert.embeddings.word_embeddings.weight.get_device()
    zeros = torch.zeros((tokens_to_add, config['decoder_hidden_size'])).to(device=device)
    encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
        (encoder.bert.embeddings.word_embeddings.weight.data, zeros)
    )

    # Size of the output vocabulary which contains the tags + begin and end
    # tokens used by the Transformer decoder.
    output_vocab_size = num_tags + 2

    decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
        d_model=config['decoder_hidden_size'],
        d_inner=config['decoder_filter_size'],
        num_layers=config['decoder_num_hidden_layers'],
        num_attn_heads=config['decoder_num_attention_heads'],
        ffn_dropout=0.1,
        vocab_size=output_vocab_size,
        attn_score_dropout=0.1,
        max_seq_length=args.max_seq_length,
        embedding_dropout=0.25,
        hidden_act='gelu',
        use_full_attention=config['use_full_attention'],
    )

    logits = nemo_nlp.nm.trainables.TokenClassifier(
        config['decoder_hidden_size'], num_classes=output_vocab_size, num_layers=1, log_softmax=False
    )

    loss_fn = CrossEntropyLossNM(logits_ndim=3)
    loss_eval_metric = CrossEntropyLossNM(logits_ndim=3, reduction='none')
    # loss_fn = nemo_nlp.nm.losses.SmoothedCrossEntropyLoss(pad_id=tokenizer.pad_id, label_smoothing=0.1)

    beam_search = nemo_nlp.nm.trainables.BeamSearchTranslatorNM(
        decoder=decoder,
        log_softmax=logits,
        max_seq_length=args.max_seq_length,
        beam_size=4,
        length_penalty=0.0,
        bos_token=tokenizer.bos_id,
        pad_token=tokenizer.pad_id,
        eos_token=tokenizer.eos_id,
    )

    # tie all embeddings weights
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

    def create_pipeline(dataset, tokenizer, num_examples, mode):

        data_layer = LaserTaggerDataLayer(dataset, tokenizer, num_examples, args.batch_size, mode == "infer")
        (
            input_ids,
            input_mask,
            segment_ids,
            tgt_ids,
            labels_mask,
            labels,
            loss_mask,
            src_ids,
            src_first_tokens,
        ) = data_layer()

        src_hiddens = encoder(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        tgt_hiddens = decoder(
            input_ids_tgt=tgt_ids, hidden_states_src=src_hiddens, input_mask_src=input_mask, input_mask_tgt=labels_mask
        )

        log_softmax = logits(hidden_states=tgt_hiddens)

        if mode != "infer":
            loss = loss_fn(logits=log_softmax, labels=labels, loss_mask=loss_mask)
            per_example_loss = loss_eval_metric(logits=log_softmax, labels=labels, loss_mask=loss_mask)

        beam_results = None
        if mode == "infer":
            beam_results = beam_search(hidden_states_src=src_hiddens, input_mask_src=input_mask)
            return [beam_results]

        return loss, [tgt_ids, input_ids, loss, per_example_loss, log_softmax, labels, labels_mask]

    if args.command == "train":
        # training pipeline
        train_loss, _ = create_pipeline(train_examples, tokenizer, num_train_examples, mode="train")

        # evaluation pipelines
        all_eval_losses = {}
        all_eval_tensors = {}
        # for eval_dataset in args.eval_datasets:
        eval_loss, eval_tensors = create_pipeline(eval_examples, tokenizer, num_eval_examples, mode="eval")
        all_eval_losses[0] = eval_loss
        all_eval_tensors[0] = eval_tensors

        def print_loss(x):
            loss = x[0].item()
            logging.info("Training loss: {:.4f}".format(loss))

        # callbacks
        callback_train = nemo.core.SimpleLossLoggerCallback(
            tensors=[train_loss],
            step_freq=100,
            print_func=print_loss,
            get_tb_values=lambda x: [["loss", x[0]]],
            tb_writer=nf.tb_writer,
        )

        callbacks = [callback_train]

        # for eval_dataset in args.eval_datasets:
        callback_eval = nemo.core.EvaluatorCallback(
            eval_tensors=all_eval_tensors[0],
            user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=args.eval_freq,
            tb_writer=nf.tb_writer,
        )
        if eval_examples:
            callbacks.append(callback_eval)

        checkpointer_callback = CheckpointCallback(folder=args.work_dir, step_freq=args.checkpoint_save_freq)
        callbacks.append(checkpointer_callback)

        max_steps, warmup_steps = _calculate_steps(num_train_examples, args.batch_size, args.num_epochs, 0.1)

        # define learning rate decay policy
        lr_policy = PolynomialDecayAnnealing(total_steps=max_steps, min_lr=0, warmup_steps=warmup_steps)

        # Create trainer and execute training action
        nf.train(
            tensors_to_optimize=[train_loss],
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
        tensors_pred = create_pipeline(test_examples, tokenizer, num_test_examples, mode="infer")
        computed_tensors = nf.infer(tensors=tensors_pred, checkpoint_dir=args.work_dir)

        id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}

        beam_results = []
        for i in computed_tensors[0]:
            beam_results.extend(i[:, 1:].cpu().numpy().tolist())

        # compute and realize predictions with LaserTagger
        sources, predictions, target_lists = [], [], []
        for i, example in enumerate(test_examples):
            example.features['labels'] = beam_results[i]
            example.features['labels_mask'] = [0] + [1] * (len(beam_results[i]) - 2) + [0]
            labels = [id_2_tag[label_id] for label_id in example.get_token_labels()]
            predictions.append(example.editing_task.realize_output(labels))

        with open(args.test_file_raw, 'r') as f:
            for line in f:
                source, *targets = line.rstrip('\n').split('\t')
                ## TODO: For bert uncased models
                # if lowercase:
                # 	source = source.lower()
                # 	pred = pred.lower()
                # 	targets = [t.lower() for t in targets]
                sources.append(source)
                target_lists.append(targets)

        exact = score_lib.compute_exact_score(predictions, target_lists)
        sari, keep, addition, deletion = score_lib.compute_sari_scores(sources, predictions, target_lists)
        logging.info(f'Exact score:     {100*exact:.3f}')
        logging.info(f'SARI score:      {100*sari:.3f}')
        logging.info(f' KEEP score:     {100*keep:.3f}')
        logging.info(f' ADDITION score: {100*addition:.3f}')
        logging.info(f' DELETION score: {100*deletion:.3f}')
