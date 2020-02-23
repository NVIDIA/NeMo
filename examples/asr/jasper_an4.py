# Copyright (c) 2019 NVIDIA Corporation
import argparse
import math
import os
from functools import partial

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
import nemo.utils.argparse as nm_argparse
from nemo.collections.asr.helpers import (
    monitor_asr_train_progress,
    post_process_predictions,
    post_process_transcripts,
    process_evaluation_batch,
    process_evaluation_epoch,
    word_error_rate,
)
from nemo.utils.lr_policies import CosineAnnealing

logging = nemo.logging


def create_dags(model_config_file, vocab, args, nf):

    # Create a data_layer for training.
    data_layer = nemo_asr.AudioToTextDataLayer.import_from_config(
        model_config_file,
        "AudioToTextDataLayer_train",
        overwrite_params={"manifest_filepath": args.train_dataset, "batch_size": args.batch_size},
    )

    num_samples = len(data_layer)
    steps_per_epoch = math.ceil(num_samples / (data_layer.batch_size * args.iter_per_step * nf.world_size))
    total_steps = steps_per_epoch * args.num_epochs
    logging.info("Train samples=", num_samples, "num_steps=", total_steps)

    # Create a data_layer for evaluation.
    data_layer_eval = nemo_asr.AudioToTextDataLayer.import_from_config(
        model_config_file, "AudioToTextDataLayer_eval", overwrite_params={"manifest_filepath": args.eval_datasets},
    )

    num_samples = len(data_layer_eval)
    logging.info(f"Eval samples={num_samples}")

    # Instantiate data processor.
    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
        model_config_file, "AudioToMelSpectrogramPreprocessor"
    )

    # Instantiate JASPER encoder-decoder modules.
    jasper_encoder = nemo_asr.JasperEncoder.import_from_config(model_config_file, "JasperEncoder")
    jasper_decoder = nemo_asr.JasperDecoderForCTC.import_from_config(
        model_config_file, "JasperDecoderForCTC", overwrite_params={"num_classes": len(vocab)}
    )

    # Instantiate losses.
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(vocab))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # Create a training graph.
    audio, audio_len, transcript, transcript_len = data_layer()
    processed, processed_len = data_preprocessor(input_signal=audio, length=audio_len)
    encoded, encoded_len = jasper_encoder(audio_signal=processed, length=processed_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,)

    # Create an evaluation graph.
    audio_e, audio_len_e, transcript_e, transcript_len_e = data_layer_eval()
    processed_e, processed_len_e = data_preprocessor(input_signal=audio_e, length=audio_len_e)
    encoded_e, encoded_len_e = jasper_encoder(audio_signal=processed_e, length=processed_len_e)
    log_probs_e = jasper_decoder(encoder_output=encoded_e)
    predictions_e = greedy_decoder(log_probs=log_probs_e)
    loss_e = ctc_loss(
        log_probs=log_probs_e, targets=transcript_e, input_length=encoded_len_e, target_length=transcript_len_e,
    )
    logging.info("Num of params in encoder: {0}".format(jasper_encoder.num_weights))

    # Callbacks to print info to console and Tensorboard.
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss, predictions, transcript, transcript_len],
        print_func=partial(monitor_asr_train_progress, labels=vocab),
        get_tb_values=lambda x: [["loss", x[0]]],
        tb_writer=nf.tb_writer,
    )

    checkpointer_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir, step_freq=args.checkpoint_save_freq)

    eval_tensors = [loss_e, predictions_e, transcript_e, transcript_len_e]
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=eval_tensors,
        user_iter_callback=partial(process_evaluation_batch, labels=vocab),
        user_epochs_done_callback=process_evaluation_epoch,
        eval_step=args.eval_freq,
        tb_writer=nf.tb_writer,
    )
    callbacks = [train_callback, checkpointer_callback, eval_callback]

    # Return entities required by the actual training.
    return (
        loss,
        eval_tensors,
        callbacks,
        total_steps,
        log_probs_e,
        encoded_len_e,
    )


def main():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()], description='AN4 ASR', conflict_handler='resolve',
    )

    # Overwrite default args
    parser.add_argument("--train_dataset", type=str, help="training dataset path")
    parser.add_argument("--eval_datasets", type=str, help="validation dataset path")

    # Create new args
    # parser.add_argument("--lm", default="./an4-lm.3gram.binary", type=str)
    parser.add_argument("--batch_size", default=48, type=int, help="size of the training batch")
    parser.add_argument("--lm", default=None, type=str)
    parser.add_argument("--test_after_training", action='store_true')
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.25, type=float)
    parser.set_defaults(
        model_config="./configs/jasper_an4.yaml",
        train_dataset="~/TestData/an4_dataset/an4_train.json",
        eval_datasets="~/TestData/an4_dataset/an4_val.json",
        work_dir="./tmp",
        optimizer="novograd",
        num_epochs=50,
        lr=0.02,
        weight_decay=0.005,
        checkpoint_save_freq=1000,
        eval_freq=100,
        amp_opt_level="O1",
    )

    args = parser.parse_args()
    betas = (args.beta1, args.beta2)

    wer_thr = 0.20
    beam_wer_thr = 0.15

    nf = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        files_to_copy=[__file__],
        optimization_level=args.amp_opt_level,
        random_seed=0,
        log_dir=args.work_dir,
        create_tb_writer=True,
        cudnn_benchmark=args.cudnn_benchmark,
    )
    tb_writer = nf.tb_writer
    checkpoint_dir = nf.checkpoint_dir

    # Load model definition
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)
    # Get vocabulary.
    vocab = jasper_params['labels']

    (loss, eval_tensors, callbacks, total_steps, log_probs_e, encoded_len_e,) = create_dags(
        args.model_config, vocab, args, nf
    )

    nf.train(
        tensors_to_optimize=[loss],
        callbacks=callbacks,
        optimizer=args.optimizer,
        lr_policy=CosineAnnealing(total_steps=total_steps, min_lr=args.lr / 100),
        optimization_params={
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "momentum": args.momentum,
            "betas": betas,
            "weight_decay": args.weight_decay,
            "grad_norm_clip": None,
        },
        batches_per_step=args.iter_per_step,
        amp_max_loss_scale=256.0,
        # synced_batchnorm=(nf.global_rank is not None),
    )

    if args.test_after_training:
        logging.info("Testing greedy and beam search with LM WER.")
        # Create BeamSearch NM
        if nf.world_size > 1 or args.lm is None:
            logging.warning("Skipping beam search WER as it does not work if doing distributed training.")
        else:
            beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
                vocab=vocab, beam_width=64, alpha=2.0, beta=1.5, lm_path=args.lm, num_cpus=max(os.cpu_count(), 1),
            )
            beam_predictions = beam_search_with_lm(log_probs=log_probs_e, log_probs_length=encoded_len_e)
            eval_tensors.append(beam_predictions)

        evaluated_tensors = nf.infer(eval_tensors)
        if nf.global_rank in [0, None]:
            greedy_hypotheses = post_process_predictions(evaluated_tensors[1], vocab)
            references = post_process_transcripts(evaluated_tensors[2], evaluated_tensors[3], vocab)
            wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
            logging.info("Greedy WER: {:.2f}%".format(wer * 100))
            if wer > wer_thr:
                nf.sync_all_processes(False)
                raise ValueError(f"Final eval greedy WER {wer * 100:.2f}% > :" f"than {wer_thr * 100:.2f}%")
        nf.sync_all_processes()

        if nf.world_size == 1 and args.lm is not None:
            beam_hypotheses = []
            # Over mini-batch
            for i in evaluated_tensors[-1]:
                # Over samples
                for j in i:
                    beam_hypotheses.append(j[0][1])

            beam_wer = word_error_rate(hypotheses=beam_hypotheses, references=references)
            logging.info("Beam WER {:.2f}%".format(beam_wer * 100))
            assert beam_wer <= beam_wer_thr, "Final eval beam WER {:.2f}%  > than {:.2f}%".format(
                beam_wer * 100, beam_wer_thr * 100
            )
            assert beam_wer <= wer, "Final eval beam WER > than the greedy WER."

        # Reload model weights and train for extra 10 epochs
        checkpointer_callback = nemo.core.CheckpointCallback(
            folder=checkpoint_dir, step_freq=args.checkpoint_save_freq, force_load=True,
        )

        # Distributed Data Parallel changes the underlying class so we need
        # to reinstantiate Encoder and Decoder
        args.num_epochs += 10
        previous_step_count = total_steps
        loss, eval_tensors, callbacks, total_steps, _, _ = create_dags(args.model_config, vocab, args, nf)

        nf.reset_trainer()
        nf.train(
            tensors_to_optimize=[loss],
            callbacks=callbacks,
            optimizer=args.optimizer,
            lr_policy=CosineAnnealing(warmup_steps=previous_step_count, total_steps=total_steps),
            optimization_params={
                "num_epochs": args.num_epochs,
                "lr": args.lr / 100,
                "momentum": args.momentum,
                "betas": betas,
                "weight_decay": args.weight_decay,
                "grad_norm_clip": None,
            },
            reset=True,
            amp_max_loss_scale=256.0,
            # synced_batchnorm=(nf.global_rank is not None),
        )

        evaluated_tensors = nf.infer(eval_tensors)
        if nf.global_rank in [0, None]:
            greedy_hypotheses = post_process_predictions(evaluated_tensors[1], vocab)
            references = post_process_transcripts(evaluated_tensors[2], evaluated_tensors[3], vocab)
            wer_new = word_error_rate(hypotheses=greedy_hypotheses, references=references)
            logging.info("New greedy WER: {:.2f}%".format(wer_new * 100))
            if wer_new > wer * 1.1:
                nf.sync_all_processes(False)
                raise ValueError(
                    f"Fine tuning: new WER {wer_new * 100:.2f}% > than the " f"previous WER {wer * 100:.2f}%"
                )
        nf.sync_all_processes()

        # Open the log file and ensure that epochs is strictly increasing
        if nf._exp_manager.log_file:
            epochs = []
            with open(nf._exp_manager.log_file, "r") as log_file:
                line = log_file.readline()
                while line:
                    index = line.find("Starting epoch")
                    if index != -1:
                        epochs.append(int(line[index + len("Starting epoch") :]))
                    line = log_file.readline()
            for i, e in enumerate(epochs):
                if i != e:
                    raise ValueError("Epochs from logfile was not understood")


if __name__ == "__main__":
    main()
