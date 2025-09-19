# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import hydra
from time import time

from nemo.collections.asr.inference.utils.manifest_io import (
    get_audio_filepaths,
    dump_output,
    calculate_duration
)
from nemo.collections.asr.inference.factory.recognizer_builder import RecognizerBuilder

from nemo.utils import logging

# disable nemo_text_processing logging
from nemo_text_processing.utils import logger as nemo_text_logger
nemo_text_logger.propagate = False


from nemo.collections.asr.inference.utils.progressbar import TQDMProgressBar

@hydra.main(version_base=None)
def main(cfg):
    # Reading audio filepaths
    audio_filepaths = get_audio_filepaths(cfg.audio_file, sort_by_duration=True)
    logging.info(f"Found {len(audio_filepaths)} audio files")

    # Build the pipeline
    recognizer = RecognizerBuilder.build_recognizer(cfg)
    progress_bar = TQDMProgressBar()

    # Run the pipeline
    start = time()
    output = recognizer.run(audio_filepaths, progress_bar=progress_bar)
    exec_dur = time() - start

    # Calculate RTFX
    data_dur = calculate_duration(audio_filepaths)
    rtfx = data_dur / exec_dur
    logging.info(f"RTFX: {rtfx:.2f} ({data_dur:.2f}s / {exec_dur:.2f}s)")

    # Dump the transcriptions to a output file
    dump_output(audio_filepaths, output, cfg.output_filename, cfg.output_ctm_dir)
    logging.info(f"Transcriptions written to {cfg.output_filename}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
