import os
import subprocess
from pathlib import Path

from processors.base_processor import BaseProcessor


class ASRInference(BaseProcessor):
    """This processor perforce ASR inference.

    Note that it does not re-use base parallel implementation, since the ASR
    inference is already run in batches.

    TODO: actually, it might still be benefitial to have another level of
        parallelization, but that needs to be tested.
    """

    def __init__(self, output_manifest_file, input_manifest_file, pretrained_model, batch_size=32):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file
        self.script_path = Path(__file__).parents[3] / "examples" / "asr" / "transcribe_speech.py"
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size

    def process(self):
        """This will add "pred_text" key into the output manifest."""
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        subprocess.run(
            f"python {self.script_path} "
            f"pretrained_name={self.pretrained_model} "
            f"dataset_manifest={self.input_manifest_file} "
            f"output_filename={self.output_manifest_file} "
            f"batch_size={self.batch_size} ",
            shell=True,
            check=True,
        )
