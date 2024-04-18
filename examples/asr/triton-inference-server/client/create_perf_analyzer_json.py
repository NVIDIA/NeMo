from datasets import load_dataset
import json
import numpy as np

import base64
import fire


def convert_dataset(dataset_name: str, tag: str, output_json: str):
    # Load the dataset
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")

    # Prepare the data for Triton Perf Analyzer
    # Assuming the model takes an input named 'AUDIO_INPUT' of type BYTES
    perf_data = {
        "data": [
            {
                "WAV":
                {
                    "content": 
                    {
                        "b64": base64.b64encode(example["audio"]["array"].astype(np.float32)).decode("utf-8"),
                    },
                    "shape": example["audio"]["array"].shape
                },
                "WAV_LENS":
                {
                    "b64": base64.b64encode(np.array([len(example["audio"]["array"])], dtype=np.int32)).decode("utf-8")
                }
            }
            for example in dataset["validation"]
        ]
    }

    # Save the JSON structure to a file
    with open(output_json, 'w') as json_file:
        json.dump(perf_data, json_file)

if __name__ == "__main__":
    fire.Fire(convert_dataset)

print("Triton Perf Analyzer JSON file has been generated.")
