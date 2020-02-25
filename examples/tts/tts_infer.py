# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
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
        "--vocoder",
        type=str,
        required=True,
        choices=["griffin-lim", "waveglow"],
        help="Vocoder used to convert from spectrograms to audio",
    )
    parser.add_argument(
        "--spec_model_config", type=str, required=True, help="spec model configuration file: model.yaml",
    )
    parser.add_argument(
        "--vocoder_model_config",
        type=str,
        help=("vocoder model configuration file: model.yaml. Not required for griffin-lim."),
    )
    parser.add_argument(
        "--spec_model_load_dir", type=str, required=True, help="directory containing checkpoints for spec model",
    )
    parser.add_argument(
        "--vocoder_model_load_dir",
        type=str,
        help=("directory containing checkpoints for vocoder model. Not required for griffin-lim"),
    )
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="directory to save audio files to")

    # Grifflin-Lim parameters
    parser.add_argument(
        "--griffin_lim_mag_scale",
        type=float,
        default=2048,
        help=(
            "This is multiplied with the linear spectrogram. This is to avoid audio sounding muted due to mel "
            "filter normalization"
        ),
    )
    parser.add_argument(
        "--griffin_lim_power",
        type=float,
        default=1.2,
        help=(
            "The linear spectrogram is raised to this power prior to running the Griffin Lim algorithm. A power of "
            "greater than 1 has been shown to improve audio quality."
        ),
    )

    # Waveglow parameters
    parser.add_argument(
        "--waveglow_denoiser_strength",
        type=float,
        default=0.0,
        help="denoiser strength for waveglow. Start with 0 and slowly increment",
    )
    parser.add_argument("--waveglow_sigma", type=float, default=0.6)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amp_opt_level", default="O1")

    args = parser.parse_args()
    if args.vocoder == "griffin-lim" and (args.vocoder_model_config or args.vocoder_model_load_dir):
        raise ValueError(
            "Griffin-Lim was specified as the vocoder but the a value for vocoder_model_config or "
            "vocoder_model_load_dir was passed."
        )
    return args


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        logging.warning("audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


def plot_and_save_spec(spectrogram, i, save_dir=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    save_file = f"spec_{i}.png"
    if save_dir:
        save_file = os.path.join(save_dir, save_file)
    plt.savefig(save_file)
    plt.close()


def create_infer_dags(
    neural_factory, neural_modules, labels, infer_dataset, infer_batch_size, cpu_per_dl=1,
):
    (_, text_embedding, t2_enc, t2_dec, t2_postnet, _, _) = neural_modules

    data_layer = nemo_asr.TranscriptDataLayer(
        path=infer_dataset,
        labels=labels,
        batch_size=infer_batch_size,
        num_workers=cpu_per_dl,
        # load_audio=False,
        bos_id=len(labels),
        eos_id=len(labels) + 1,
        pad_id=len(labels) + 2,
        shuffle=False,
    )
    transcript, transcript_len = data_layer()

    transcript_embedded = text_embedding(char_phone=transcript)
    transcript_encoded = t2_enc(char_phone_embeddings=transcript_embedded, embedding_length=transcript_len,)
    if isinstance(t2_dec, nemo_tts.Tacotron2DecoderInfer):
        mel_decoder, gate, alignments, mel_len = t2_dec(
            char_phone_encoded=transcript_encoded, encoded_length=transcript_len,
        )
    else:
        raise ValueError("The Neural Module for tacotron2 decoder was not understood")
    mel_postnet = t2_postnet(mel_input=mel_decoder)

    return [mel_postnet, gate, alignments, mel_len]


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
            labels = tacotron2_params["labels"]
        spec_neural_modules = create_NMs(args.spec_model_config, labels=labels, decoder_infer=True)
        infer_tensors = create_infer_dags(
            neural_factory=neural_factory,
            neural_modules=spec_neural_modules,
            labels=labels,
            infer_dataset=args.eval_dataset,
            infer_batch_size=args.batch_size,
        )

    logging.info("Running Tacotron 2")
    # Run tacotron 2
    evaluated_tensors = neural_factory.infer(
        tensors=infer_tensors, checkpoint_dir=args.spec_model_load_dir, cache=use_cache, offload_to_cpu=False,
    )
    mel_len = evaluated_tensors[-1]
    logging.info("Done Running Tacotron 2")
    filterbank = librosa.filters.mel(
        sr=tacotron2_params["sample_rate"],
        n_fft=tacotron2_params["n_fft"],
        n_mels=tacotron2_params["n_mels"],
        fmax=tacotron2_params["fmax"],
    )

    if args.vocoder == "griffin-lim":
        logging.info("Running Griffin-Lim")
        mel_spec = evaluated_tensors[0]
        for i, batch in enumerate(mel_spec):
            log_mel = batch.cpu().numpy().transpose(0, 2, 1)
            mel = np.exp(log_mel)
            magnitudes = np.dot(mel, filterbank) * args.griffin_lim_mag_scale
            for j, sample in enumerate(magnitudes):
                sample = sample[: mel_len[i][j], :]
                audio = griffin_lim(sample.T ** args.griffin_lim_power)
                save_file = f"sample_{i * 32 + j}.wav"
                if args.save_dir:
                    save_file = os.path.join(args.save_dir, save_file)
                write(save_file, tacotron2_params["sample_rate"], audio)
                plot_and_save_spec(log_mel[j][: mel_len[i][j], :].T, i * 32 + j, args.save_dir)

    elif args.vocoder == "waveglow":
        (mel_pred, _, _, _) = infer_tensors
        if not args.vocoder_model_config or not args.vocoder_model_load_dir:
            raise ValueError(
                "Using waveglow as the vocoder requires the "
                "--vocoder_model_config and --vocoder_model_load_dir args"
            )

        yaml = YAML(typ="safe")
        with open(args.vocoder_model_config) as file:
            waveglow_params = yaml.load(file)
        waveglow = nemo_tts.WaveGlowInferNM(sigma=args.waveglow_sigma, **waveglow_params["WaveGlowNM"])
        audio_pred = waveglow(mel_spectrogram=mel_pred)
        # waveglow.restore_from(args.vocoder_model_load_dir)

        # Run waveglow
        logging.info("Running Waveglow")
        evaluated_tensors = neural_factory.infer(
            tensors=[audio_pred],
            checkpoint_dir=args.vocoder_model_load_dir,
            # checkpoint_dir=None,
            modules_to_restore=[waveglow],
            use_cache=use_cache,
        )
        logging.info("Done Running Waveglow")

        if args.waveglow_denoiser_strength > 0:
            logging.info("Setup denoiser")
            waveglow.setup_denoiser()

        logging.info("Saving results to disk")
        for i, batch in enumerate(evaluated_tensors[0]):
            audio = batch.cpu().numpy()
            for j, sample in enumerate(audio):
                sample_len = mel_len[i][j] * tacotron2_params["n_stride"]
                sample = sample[:sample_len]
                save_file = f"sample_{i * 32 + j}.wav"
                if args.save_dir:
                    save_file = os.path.join(args.save_dir, save_file)
                if args.waveglow_denoiser_strength > 0:
                    sample, spec = waveglow.denoise(sample, strength=args.waveglow_denoiser_strength)
                else:
                    spec, _ = librosa.core.magphase(librosa.core.stft(sample, n_fft=waveglow_params["n_fft"]))
                write(save_file, waveglow_params["sample_rate"], sample)
                spec = np.dot(filterbank, spec)
                spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))
                plot_and_save_spec(spec, i * 32 + j, args.save_dir)


if __name__ == '__main__':
    main()
