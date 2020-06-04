from nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm import BERT
import argparse
import json
import os
import math

import numpy as np

import nemo
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.tokenizers.tokenizer_utils
import nemo.core as nemo_core
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.utils.lr_policies import get_lr_policy

import bert_example
import tagging_converter
import utils
import torch

from absl import logging

from nemo.collections.nlp.callbacks.machine_translation_callback import (
	eval_epochs_done_callback_wer,
	eval_iter_callback,
)
from nemo.core import WeightShareTransform
from nemo.core.callbacks import CheckpointCallback
from nemo.utils.lr_policies import SquareAnnealing
from lasertagger_datalayer import *

def parse_args():
	parser = nemo.utils.NemoArgParser(description='LaserTagger')
	parser.set_defaults(
		optimizer="adam_w",
		amp_opt_level="O1",
		num_epochs=3,
		batch_size=64,
		eval_batch_size=8,
		lr=3e-5,
		weight_decay=0,
		max_steps=2000,
		iter_per_step=1,
		checkpoint_save_freq=1000,
		work_dir='outputs/lt-2',
		eval_freq=200,
	)
	parser.add_argument(
		"--train_file", type=str, help="The path to training pkl file",
	)
	parser.add_argument(
		"--eval_file", type=str, help="The path to evaluation pkl file",
	)
	parser.add_argument(
		"--label_map_file",
		type=str, 
		help="Path to the label map file. Either a JSON file ending with '.json', \
				that maps each possible tag to an ID, or a text file that \
				has one tag per line."
	)
	parser.add_argument(
		'--pretrained_model_name',
		default='bert-base-cased',
		type=str,
		help='Name of the pre-trained model',
		choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
	)
	parser.add_argument(
		'--bert_checkpoint',
		default='bert-base-cased',
		type=str,
		help='Path to the pre-trained model checkpoint'
	)
	parser.add_argument("--model_config_file", default=None, type=str, help="Path to LaserTagger config file in json format")
	parser.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
	parser.add_argument("--max_seq_length", default=128, type=int)
	parser.add_argument("--warmup_steps", default=4500, type=int)

	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()

	num_tags = len(utils.read_label_map(args.label_map_file))

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

	tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)
	vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
	tokens_to_add = vocab_size - tokenizer.vocab_size

	encoder = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_model_name)

	device = encoder.bert.embeddings.word_embeddings.weight.get_device()
	zeros = torch.zeros((tokens_to_add, config['hidden_size'])).to(device=device)
	encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
		(encoder.bert.embeddings.word_embeddings.weight.data, zeros)
	)

	# hidden_size = model.hidden_size

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
		hidden_act='gelu'
	)

	logits = nemo_nlp.nm.trainables.TokenClassifier(
		config['decoder_hidden_size'], num_classes=output_vocab_size, num_layers=1, log_softmax=True
	)

	loss_fn = CrossEntropyLossNM(logits_ndim=3)
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
			"embedding_layer.token_embedding.weight": ("bert.embeddings.word_embeddings.weight", WeightShareTransform.SAME)
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

	def create_pipeline(dataset, training=True):

		data_layer = LaserTaggerDataLayer(preprocessed_data=dataset)
		input_ids, input_mask, segment_ids, tgt_ids, labels_mask, labels = data_layer()

		src_hiddens = encoder(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
		tgt_hiddens = decoder(
			input_ids_tgt=tgt_ids, hidden_states_src=src_hiddens, input_mask_src=input_mask, input_mask_tgt=labels_mask
		)

		log_softmax = logits(hidden_states=tgt_hiddens)
		loss = loss_fn(logits=log_softmax, labels=labels)
		beam_results = None
		if not training:
			beam_results = beam_search(hidden_states_src=src_hiddens, input_mask_src=input_mask)
		return loss, [tgt_ids, loss, beam_results]
	
	# training pipeline
	train_loss, _ = create_pipeline(args.train_file)

	# evaluation pipelines
	all_eval_losses = {}
	all_eval_tensors = {}
	# for eval_dataset in args.eval_datasets:
	eval_loss, eval_tensors = create_pipeline(args.eval_file, training=False)
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
		user_epochs_done_callback=eval_epochs_done_callback_wer,
		eval_step=args.eval_freq,
		tb_writer=nf.tb_writer,
	)
	# callbacks.append(callback_eval)

	checkpointer_callback = CheckpointCallback(folder=args.work_dir, step_freq=args.checkpoint_save_freq)
	callbacks.append(checkpointer_callback)

	# define learning rate decay policy
	lr_policy = SquareAnnealing(total_steps=args.max_steps, min_lr=1e-5, warmup_steps=args.warmup_steps)

	# Create trainer and execute training action
	nf.train(
		tensors_to_optimize=[train_loss],
		callbacks=callbacks,
		optimizer=args.optimizer,
		lr_policy=lr_policy,
		optimization_params={
			"num_epochs": 300,
			"max_steps": args.max_steps,
			"lr": args.lr,
			"weight_decay": args.weight_decay,
		},
		batches_per_step=args.iter_per_step,
	)
