from nemo.collections import llm
from nemo.collections.llm.gpt.model.ssm import HFNemotronHExporter
from nemo.collections.llm.gpt.model.ssm import HFNemotronHImporter
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion_type", type=str, required=True)
    parser.add_argument("--source_ckpt", type=str, required=True)
    parser.add_argument("--target_ckpt", type=str, required=True)
    return parser.parse_args()



if __name__ == "__main__":

    args = get_args()
    nmh_config = llm.NemotronHConfig4B()
    if args.conversion_type == "NEMO2_TO_HF":
            
        exporter = HFNemotronHExporter(args.source_ckpt, model_config=nmh_config)
        exporter.apply(args.target_ckpt)

    elif args.conversion_type == "HF_TO_NEMO2":

        exporter = HFNemotronHImporter(args.source_ckpt)
        exporter.apply(args.target_ckpt)
    
    else:
        raise ValueError(f"Invalid conversion type: {args.conversion_type}")