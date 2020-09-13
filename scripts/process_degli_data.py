import argparse
import sys

from nemo.collections.tts.data.datalayers import DegliProprocssing

sys.path.insert(0, '../')


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset fitted for training and validating deep griffin iteration from wavefiles'
    )
    parser.add_argument(
        "-v",
        "--valid_filelist",
        help="Filelist for validation set, with all validation audio files listed",
        required=True,
        default=None,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--train_filelist",
        help="Filelist for train set, with all train audio files listed",
        required=True,
        default=None,
        type=str,
    )
    parser.add_argument(
        "-n",
        "--n_fft",
        help="Value for the n_fft parameter, and the filter length for the STFT",
        default=512,
        type=int,
    )
    parser.add_argument("--hop_length", help="STFT parameter", default=256, type=int)
    parser.add_argument(
        "-d", "--destination", help="Destination to save the preprocessed data set to", default="/tmp", type=str
    )
    parser.add_argument(
        "-s",
        "--num_snr",
        help="Number of distinctive noisy samples to generate for each clear sample at the file list",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    DegliProprocssing(
        args.valid_filelist, args.train_filelist, args.n_fft, args.hop_length, args.num_snr, args.destination
    )


if __name__ == "__main__":
    main()
