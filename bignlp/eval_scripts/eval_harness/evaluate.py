import argparse
import json
import glob
from omegaconf import OmegaConf
from typing import Union
import random
import logging
import sys
import os
import string
import shutil
import time
from datetime import datetime
import subprocess

from lm_eval import models, tasks, evaluator, base, utils

logger = logging.getLogger('__main__')

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


def parse_args(parser_main):
    # parser = argparse.ArgumentParser()
    parser = parser_main.add_argument_group(title='evaluate-tasks')
    # Experiment
    parser.add_argument('--name', dest='experiment_name', type=str, default='',
                        help='A string identifier/name for the experiment to be run '
                             '- it will be appended to the output directory name, before the timestamp')
    parser.add_argument('--comment', type=str, default='', help='An optional comment/description of the experiment. '
                                                                'Will be included in the configuration JSON file and '
                                                                'potentially other output files.')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='If set, a timestamp and random suffix will NOT be appended to the output directory name '
                             '(unless no `name` is specified)')
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--cache_dir', default="")
    parser.add_argument('--eval_seed', type=int, default=1234,
                        help='Random seed used for python, numpy, [pytorch, and cuda.]')
    parser.add_argument('--limit', type=int, default=None,
                        help='If specified, will limit evaluation set to this number of samples')
    # I/O
    parser.add_argument('--output_path', type=str, default='.',
                        help="Path to output root directory. Must exist. An experiment directory containing metrics, "
                             "predictions, configuration etc will be created inside.")
    parser.add_argument('--serialize_predictions', action="store_true",
                        help="If set, the model's predictions (and other information) will be serialized to disk.")
    # H/W configuration
    parser.add_argument('--batch_size', type=int, default=1)
    # Warning: cuda device is the only way it could work.
    parser.add_argument('--device', type=str, default='cuda')
    # Model
    parser.add_argument('--model', required=True)

    parser.add_argument("--nemo_model", type=str, default=None, required=False, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Path to PTL checkpoints saved during training. Ex: "
             "/raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Name of checkpoint to be used. Ex: "
             "megatron_gpt--val_loss=6.34-step=649-last.ckpt",
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Path to config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")

    parser.add_argument('--vocab_file', default=None)
    parser.add_argument('--merge_file', default=None)

    # Prompt
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument("--filter_shots", action='store_true',
                        help="Filter examples used as shots in the prompt, "
                             "e.g. exclude examples of the same type as the sample under evaluation.")
    # HANS
    parser.add_argument("--ratio_positive", type=float, default=None,
                        help='Ratio of examples with a positive label')
    parser.add_argument("--mix_mode", type=str, default='shuffle',
                        choices=['shuffle', 'interleave_first', 'interleave_last', 'pos_first', 'neg_first'],
                        help='How to mix (arrange order of) positive and negative shot examples in the prompt')
    parser.add_argument("--interleave_width", type=int, default=1,
                        help='The number of consecutive examples with the same label, when `mix_mode` is interleave')

    # Generation tasks
    parser.add_argument("--generate-max-token", type=int, default=0, help='Max tokens to generate.')

    # return parser.parse_args()
    return parser_main


def can_write_output(lm: Union[base.CachingLM, base.LM], args: argparse.Namespace) -> bool:
    """Only 1 process should print and dump results, this function would only return True
       for 1 of the processes that has full output
    """
    if isinstance(lm, base.CachingLM):
        return True
    elif lm.can_access_output():
        return True
    else:
        return False


def setup_output_dir(args, local_args=None, unknown_args=None):
    """Prepare experiment output directories and save configuration.
    Will UPDATE args
    Input:
        args: arguments object from argparse. Contains all arguments
        local_args: arguments object from argparse, containing only arguments recognized by the parser of this script.
            If specified, will also write these local arguments to a separate JSON file.
        unknown_args: arguments object from argparse, containing only arguments NOT recognized by the parser of this script.
            If specified, will also write these arguments to the local arguments JSON file.
    Returns:
        config: configuration dictionary
    """

    # Create output directory
    initial_timestamp = datetime.now()
    output_path = args.output_path
    if not os.path.isdir(output_path):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                output_path))

    output_path = os.path.join(output_path, args.experiment_name)

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    args.initial_timestamp = formatted_timestamp
    if (not args.no_timestamp) or (len(args.experiment_name) == 0):
        random.seed(args.eval_seed)
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_path += "_" + formatted_timestamp + "_" + rand_suffix
        # random.seed(args.eval_seed)
    args.output_path = output_path
    args.pred_dir = os.path.join(output_path, 'predictions')
    utils.create_dirs([args.pred_dir])

    # Add file logging besides stdout
    # file_handler = logging.FileHandler(os.path.join(args.output_path, 'output.log'))
    # logger.addHandler(file_handler)

    logger.info('Command:\n{}'.format(' '.join(sys.argv)))  # command used to run program

    # Save configuration as a (pretty) json file
    config = args.__dict__
    # # TODO: Raises an error, because some megatron arguments are non-serializable 
    # with open(os.path.join(output_path, 'full_configuration.json'), 'w') as fp:
    #     json.dump(config, fp, indent=4)

    with open(os.path.join(output_path, 'command.txt'), 'w') as fp:
        fp.write(' '.join(sys.argv))  # command used to run program
        fp.write("\nUnknown args: {}".format(unknown_args))

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                           cwd=os.path.dirname(os.path.abspath(__file__))).decode()
        git_diff = subprocess.check_output(['git', 'diff'], cwd=os.path.dirname(os.path.abspath(__file__))).decode()
        with open(os.path.join(output_path, 'git.txt'), 'w') as fp:
            fp.write("Git hash: {}\n".format(git_hash))
            fp.write(git_diff)
        logger.info("Git hash: {}".format(git_hash))
    except Exception as x:
        logger.error("git version not found")
        # raise x

    if local_args:
        local_config = local_args.__dict__  # local configuration dictionary
        # update local configuration with the actual values used in the global configuration
        for opt in local_config:
            if opt in config:
                local_config[opt] = config[opt]
        with open(os.path.join(output_path, 'eval_configuration.json'), 'w') as fp:
            json.dump(local_config, fp, indent=4)

    logger.info("Stored configuration file(s) in '{}'".format(output_path))

    return args


def _inject_model_parallel_rank(filepath, tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    """
    Injects tensor/pipeline model parallel ranks into the filepath.
    Does nothing if not using model parallelism.
    """
    tensor_model_parallel_rank = pipeline_model_parallel_rank = 0
    if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
        # filepath needs to be updated to include mp_rank
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
            filepath = f'{dirname}/mp_rank_{tensor_model_parallel_rank:02d}/{basename}'
        else:
            filepath = f'{dirname}/tp_rank_{tensor_model_parallel_rank:02d}_pp_rank_{pipeline_model_parallel_rank:03d}/{basename}'
        return filepath
    else:
        return filepath


def main():
    total_start_time = time.time()

    parser = argparse.ArgumentParser()
    eval_args, unknown_args = parse_args(parser).parse_known_args()
    args = eval_args

    assert args is not None
    if "nemo-gpt3" in args.model:
        assert args.device == 'cuda', "devices == 'cuda' are required to run nemo evaluations."

    if args.model == "nemo-gpt3":
        args.model = "nemo-gpt3-tp-pp"

    checkpoint_folder = args.checkpoint_folder
    checkpoint_name = args.checkpoint_name
    hparams_file = args.hparams_file
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    vocab_file = args.vocab_file
    merge_file = args.merge_file

    # Checkpoint search
    if checkpoint_name == "latest":
        checkpoints = os.path.join(checkpoint_folder, '*.ckpt')
        checkpoints = _inject_model_parallel_rank(checkpoints, tensor_model_parallel_size, pipeline_model_parallel_size)
        checkpoint_list = glob.glob(checkpoints)
        latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
        checkpoint_name = os.path.basename(latest_checkpoint)

    checkpoint = os.path.join(checkpoint_folder, checkpoint_name)
    checkpoint = _inject_model_parallel_rank(checkpoint, tensor_model_parallel_size, pipeline_model_parallel_size)
    checkpoint_list = glob.glob(checkpoint)
    if len(checkpoint_list) > 1:
        raise ValueError("Too many checkpoints fit the checkpoint name pattern in conversion config.")
    if len(checkpoint_list) == 0:
        raise ValueError("No checkpoint found with the checkpoint name pattern in conversion config.")
    args.checkpoint_name = os.path.basename(checkpoint_list[0])

    # Create hparam override file for vocab and merge
    hparams_override_file = None
    if hparams_file is not None:
        hparams_override_file = os.path.join(args.output_path, "hparams_override.yaml")
        conf = OmegaConf.load(hparams_file)
        if vocab_file is not None:
            conf.cfg.tokenizer.vocab_file = vocab_file
        if merge_file is not None:
            conf.cfg.tokenizer.merge_file = merge_file

        if is_global_rank_zero():
            with open(hparams_override_file, 'w') as f:
                OmegaConf.save(config=conf, f=f)

        while not os.path.exists(hparams_override_file):
            time.sleep(1)
    args.hparams_file = hparams_override_file

    lm = models.get_model(args.model)(args, batch_size=args.batch_size)

    # Determine whether process is allowed to write to disk
    # can_write_output() limits the processes allowed to enter clause
    write_permission = can_write_output(lm, args)

    if write_permission:
        args = setup_output_dir(args, eval_args, unknown_args)

        if args.limit:
            logger.warning("At most {} samples will be used. --limit SHOULD ONLY BE USED FOR TESTING. "
                           "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.".format(args.limit))

        if args.filter_shots:
            logger.info("Few-shot example shots will be filtered")
        else:
            logger.info("Few-shot example shots will NOT be filtered")

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names, args.cache_dir)

    if args.serialize_predictions:
        no_serialization = [name for name, task in task_dict.items() if not hasattr(task, "serialize_results")]
        if len(no_serialization):  # Only serialize for those that have implemented the method, instead of raising exception
            logger.error(
                f"Model outputs for {no_serialization} task(s) will not be dumped. Please check the implementation of {no_serialization} to "
                f"make sure you have implemented serialize_results.")
            raise Exception(
                f"args.serialize_predictions is set for dumping results, but tasks: {no_serialization} do not implement the serialize_results method")

    utils.set_seed(args.eval_seed)
    results = evaluator.evaluate(lm, task_dict, args.provide_description, args.num_fewshot, args.limit,
                                 filter_shots=args.filter_shots, serialize_predictions=args.serialize_predictions,
                                 ratio_positive=args.ratio_positive, mix_mode=args.mix_mode,
                                 interleave_width=args.interleave_width)

    if write_permission:
        summary = json.dumps(results["results"], indent=2)
        logger.info('\n' + summary)
        with open(os.path.join(args.output_path, 'metrics.json'), mode='w') as fp:
            fp.write(summary)

        if ("output" in results):
            # TODO(GEO): maybe add a for loop over "taskX" in results['output'][taskX] to store each task separately
            # Store predictions, prompts, labels etc per document as a (pretty) json file
            predictions_filepath = os.path.join(args.pred_dir, args.experiment_name + "_predictions.json")
            logger.info("Stored predictions in '{}'".format(predictions_filepath))
            with open(predictions_filepath, mode='w') as fp:
                json.dump(results, fp, indent=4)

        # MAKE TABLE
        from pytablewriter import MarkdownTableWriter, LatexTableWriter

        md_writer = MarkdownTableWriter()
        latex_writer = LatexTableWriter()
        md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
        latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

        values = []

        for k, dic in results["results"].items():
            version = results["versions"][k]
            for m, v in dic.items():
                if m.endswith("_stderr"): continue

                if m + "_stderr" in dic:
                    se = dic[m + "_stderr"]

                    values.append([k, version, m, '%.4f' % v, '±', '%.4f' % se])
                else:
                    values.append([k, version, m, '%.4f' % v, '', ''])
                k = ""
                version = ""
        md_writer.value_matrix = values
        latex_writer.value_matrix = values

        if hparams_override_file is not None:
            os.rename(hparams_override_file, os.path.join(args.output_path, "hparams_override.yaml"))

        logger.info(
            f"{args.model}, limit: {args.limit}, provide_description: {args.provide_description}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}")
        logger.info('\n' + md_writer.dumps())

        total_time = time.time() - total_start_time
        logger.info("Total runtime: {} hours, {} minutes, {} seconds".format(*utils.readable_time(total_time)))
        logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
