import argparse

from nemo.utils import logging
from nemo.collections.llm import deploy

def get_args():
    parser = argparse.ArgumentParser(description='Test evaluation with lm-eval-harness on nemo2 model deployed on PyTriton')
    parser.add_argument('--nemo2_ckpt_path', type=str, help="NeMo 2.0 ckpt path")
    parser.add_argument('--max_batch_size', type=int, help="Max BS for the model")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    try:
        deploy(
            nemo_checkpoint=args.nemo2_ckpt_path,
            max_batch_size=args.max_batch_size,
        )
    except Exception as e:
        logging.error(f"Deploy process encountered an error: {e}")
    logging.info("Deploy process terminated.")