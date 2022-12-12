"""
Example usage:
 python3 auto_blend.py \
    model_type=<mt5/t5/gpt3> \
    preprocessed_dir=<path/to/preprocessed_dir> \
    blending_alpha=<blending_alpha>
"""

import os
import math
import hydra
from collections import defaultdict

@hydra.main(config_path="conf", config_name="auto_blend")
def generate_data_blend(cfg):
    """
    Generate data blend as NeMo input `model.data.data_prefix` for binary dataset files
    within the dataset folder based on the file sizes.
    """
    model_type = cfg.get("model_type")
    data_dir = cfg.get("preprocessed_dir")
    alpha = cfg.get("blending_alpha")

    data_files = os.listdir(data_dir)
    split_size = defaultdict(int)
    file_size = defaultdict(list)
    for f in data_files:
        if f.endswith(".bin"):
            f_path = os.path.join(data_dir, f)
            f_size = os.path.getsize(f_path)
            if model_type == "mt5":
                elements = f.split("_")
                split = elements[0]
            else:
                split = f_path.strip(".bin")
            split_size[split] += f_size
            file_size[split].append((f_path.strip(".bin"), f_size))

    split_ratio = {split: math.pow(split_size[split], alpha) for split in split_size}
    total = sum(split_ratio.values())
    split_ratio = {split: split_ratio[split] / total for split in split_ratio}

    res = []
    for split in file_size:
        for prefix, size in file_size[split]:
            res.extend([round(size / split_size[split] * split_ratio[split], 6), prefix])

    print(str(res).replace(" ", ""))

if __name__ == "__main__":
    generate_data_blend()