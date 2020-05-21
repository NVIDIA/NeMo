from nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm import BERT
import argparse
import json
import os

import numpy as np

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.tokenizers.tokenizer_utils
import nemo.core as nemo_core
from nemo import logging
from nemo.collections.nlp.callbacks.qa_squad_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.utils.lr_policies import get_lr_policy

import bert_example
import tagging_converter
import utils

from absl import logging

def parse_args():
	parser = argparse.ArgumentParser(description="Squad_with_pretrained_BERT")
	parser.add_argument(
		"--train_file", type=str, help="The training data file. Should be *.json",
	)
	parser.add_argument(
		"--eval_file", type=str, help="The evaluation data file. Should be *.json",
	)
	parser.add_argument(
		"--test_file", type=str, help="The test data file. Should be *.json. Does not need to contain ground truth",
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
	parser.add_argument("--num_train_examples", default=2, type=int, help="Total number of training examples.")
	parser.add_argument("--num_eval_examples", default=-1, type=int, help="Total number of evaluation examples.")
	parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
	parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")


def read_input_file(args, input_file):

  label_map = utils.read_label_map(args.label_map_file)
  converter = tagging_converter.TaggingConverter(
	  tagging_converter.get_phrase_vocabulary_from_label_map(label_map), True)
  builder = bert_example.BertExampleBuilder(label_map, args.vocab_file,
			  args.max_seq_length, False, converter)

  num_converted = 0
  for i, (sources, target) in enumerate(utils.yield_sources_and_targets(input_file)):
	  logging.log_every_n(
		  logging.INFO,
		  f'{i} examples processed, {num_converted} converted to tf.Example.',
		  10000)
	  example = builder.build_bert_example(
		  sources, target,
		  True) # Set True to output arbitrary targets for infeasible examples
	  if example is None:
		continue
	  num_converted += 1
  logging.info(f'Done. {num_converted} examples converted to tf.Example.')
  return example, num_converted

if __name__ == "__main__":
  
  args = parse_args()

  train_examples, num_train_examples = read_input_file(args, args.train_file)
  eval_examples, num_eval_examples = read_input_file(args, args.eval_file)

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

  model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
	pretrained_model_name=args.pretrained_model_name,
	config=config,
	vocab=args.vocab_file,
	checkpoint=args.bert_checkpoint,
  )

  # hidden_size = model.hidden_size

  # Size of the output vocabulary which contains the tags + begin and end
  # tokens used by the Transformer decoder.
  output_vocab_size = num_tags + 2

  decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
	d_model=config.decoder_hidden_size,
	d_inner=config.decoder_filter_size,
	num_layers=config.decoder_num_hidden_layers,
	num_attn_heads=config.decoder_num_attention_heads,
	ffn_dropout=0.1,
	vocab_size=output_vocab_size,
	attn_score_dropout=0.1,
	max_seq_length=args.max_seq_length
  )

  logits = nemo_nlp.nm.trainables.TokenClassifier(
	config.decoder_hidden_size, num_classes=output_vocab_size, num_layers=1, log_softmax=True
  )

  def create_pipeline(dataset_src, dataset_tgt, tokens_in_batch, clean=False, training=True):
	
	input_ids = 

	tgt_hiddens = decoder(
	  input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
	)
	logits = log_softmax(hidden_states=tgt_hiddens)
	loss = loss_fn(logits=logits, labels=labels)
	beam_results = None
	if not training:
		beam_results = beam_search(hidden_states_src=src_hiddens, input_mask_src=src_mask)
	return loss, [tgt, loss, beam_results, sent_ids]



