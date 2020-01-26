# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import os
import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from ruamel.yaml import YAML
from scipy.io.wavfile import write
from tacotron2 import create_NMs

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts

logging = nemo.logging


def parse_args():
    parser = argparse.ArgumentParser(description='TTS')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument(
        "--spec_model",
        type=str,
        required=True,
        choices=["tacotron2"],
        help="Model generated to generate spectrograms",
    )
    parser.add_argument(
        "--spec_model_config", type=str, required=True, help="spec model configuration file: model.yaml",
    )
    parser.add_argument(
        "--vocoder_model_config",
        type=str,
        help=("vocoder model configuration file: model.yaml. Not required for " "griffin-lim."),
    )
    parser.add_argument(
        "--spec_model_load_dir", type=str, required=True, help="directory containing checkpoints for spec model",
    )
    parser.add_argument("--eval_dataset", type=str, required=True)

    # Grifflin-Lim parameters
    parser.add_argument(
        "--griffin_lim_mag_scale",
        type=float,
        default=2048,
        help=(
            "This is multiplied with the linear spectrogram. This is "
            "to avoid audio sounding muted due to mel filter normalization"
        ),
    )
    parser.add_argument(
        "--griffin_lim_power",
        type=float,
        default=1.2,
        help=(
            "The linear spectrogram is raised to this power prior to running"
            "the Griffin Lim algorithm. A power of greater than 1 has been "
            "shown to improve audio quality."
        ),
    )
    parser.add_argument(
        '--durations_dir', type=str, default='durs',
    )

    # Waveglow parameters
    parser.add_argument(
        "--waveglow_denoiser_strength",
        type=float,
        default=0.0,
        help=("denoiser strength for waveglow. Start with 0 and slowly " "increment"),
    )
    parser.add_argument("--waveglow_sigma", type=float, default=0.6)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--amp_opt_level", default="O1")

    args = parser.parse_args()

    return args


def create_infer_dags(
    neural_factory,
    neural_modules,
    tacotron2_config_file,
    tacotron2_params,
    infer_dataset,
    infer_batch_size,
    labels,
    cpu_per_dl=1,
):
    (data_preprocessor, text_embedding, t2_enc, t2_dec, t2_postnet, _, _) = neural_modules

    data_layer = nemo_asr.AudioToTextDataLayer.import_from_config(
        tacotron2_config_file,
        "AudioToTextDataLayer_eval",
        overwrite_params={
            "manifest_filepath": infer_dataset,
            "batch_size": infer_batch_size,
            "num_workers": cpu_per_dl,
            "bos_id": len(labels),
            "eos_id": len(labels) + 1,
            "pad_id": len(labels) + 2,
        },
    )

    audio, audio_len, transcript, transcript_len = data_layer()
    spec_target, spec_target_len = data_preprocessor(input_signal=audio, length=audio_len)

    transcript_embedded = text_embedding(char_phone=transcript)
    transcript_encoded = t2_enc(char_phone_embeddings=transcript_embedded, embedding_length=transcript_len,)
    if isinstance(t2_dec, nemo_tts.Tacotron2Decoder):
        t2_dec.force = True
        mel_decoder, gate, alignments = t2_dec(
            char_phone_encoded=transcript_encoded, encoded_length=transcript_len, mel_target=spec_target,
        )
    else:
        raise ValueError("The Neural Module for tacotron2 decoder was not understood")
    mel_postnet = t2_postnet(mel_input=mel_decoder)

    return [mel_postnet, gate, alignments, spec_target_len, transcript_len]


def main():
    args = parse_args()
    neural_factory = nemo.core.NeuralModuleFactory(
        optimization_level=args.amp_opt_level, backend=nemo.core.Backend.PyTorch, local_rank=args.local_rank,
    )

    use_cache = True
    if args.local_rank is not None:
        logging.info("Doing ALL GPU")
        use_cache = False

    # Create text to spectrogram model
    if args.spec_model == "tacotron2":
        yaml = YAML(typ="safe")
        with open(args.spec_model_config) as file:
            tacotron2_params = yaml.load(file)
        spec_neural_modules = create_NMs(
            args.spec_model_config, labels=tacotron2_params['labels'], decoder_infer=False
        )
        infer_tensors = create_infer_dags(
            neural_factory=neural_factory,
            neural_modules=spec_neural_modules,
            tacotron2_config_file=args.spec_model_config,
            tacotron2_params=tacotron2_params,
            infer_dataset=args.eval_dataset,
            infer_batch_size=args.batch_size,
            labels=tacotron2_params['labels'],
        )

    logging.info("Running Tacotron 2")
    # Run tacotron 2
    evaluated_tensors = neural_factory.infer(
        tensors=infer_tensors, checkpoint_dir=args.spec_model_load_dir, cache=False, offload_to_cpu=True,
    )

    def get_D(alignment, true_len):
        D = np.array([0 for _ in range(np.shape(alignment)[1])])

        for i in range(np.shape(alignment)[0]):
            max_index = alignment[i].tolist().index(alignment[i].max())
            D[max_index] = D[max_index] + 1

        assert D.sum() == alignment.shape[0]
        assert D.sum() == true_len

        return D

    # Save durations.
    alignments_dir = pathlib.Path(args.durations_dir)
    alignments_dir.mkdir(exist_ok=True)
    k = -1
    for alignments, mel_lens, text_lens in zip(
        tqdm.tqdm(evaluated_tensors[2]), evaluated_tensors[3], evaluated_tensors[4],
    ):
        for alignment, mel_len, text_len in zip(alignments, mel_lens, text_lens):
            alignment = alignment.cpu().numpy()
            mel_len = mel_len.cpu().numpy().item()
            text_len = text_len.cpu().numpy().item()
            dur = get_D(alignment[:mel_len, :text_len], mel_len)
            k += 1
            np.save(alignments_dir / f'{k}.npy', dur, allow_pickle=False)


if __name__ == '__main__':
    main()
