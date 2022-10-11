import os
import tempfile
import uuid

import hydra
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    processors_to_run = cfg.get("processors_to_run", "all")

    if processors_to_run == "all":
        processors_to_run = ":"
    # converting processors_to_run into Python slice
    processors_to_run = slice(*map(lambda x: int(x.strip()) if x.strip() else None, processors_to_run.split(":")))
    processors_cfgs = cfg.processors[processors_to_run]
    logging.info(
        "Specified to run the following processors: %s ", [cfg["_target_"] for cfg in processors_cfgs],
    )

    processors = []
    # let's build all processors first to automatically check
    # for errors in parameters
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, processor_cfg in enumerate(processors_cfgs):
            logging.info('=> Building processor "%s"', processor_cfg["_target_"])
            # we are assuming that each processor defines "output_manifest_file"
            # and "input_manifest_file" keys, which can be optional. In case they
            # are missing, we are creating tmp files here for them
            if "output_manifest_file" not in processor_cfg:
                tmp_file_path = os.path.join(tmp_dir, str(uuid.uuid4()))
                processor_cfg["output_manifest_file"] = tmp_file_path
                if idx != len(processors_cfgs) - 1 and "input_manifest_file" not in processors_cfgs[idx + 1]:
                    processors_cfgs[idx + 1]["input_manifest_file"] = tmp_file_path
            processor = hydra.utils.instantiate(processor_cfg)
            # running runtime tests to fail right-away if something is not
            # matching users expectations
            processor.test()
            processors.append(processor)

        for processor in processors:
            # TODO: add proper str method to all classes for good display
            logging.info('=> Running processor "%s"', processor)
            processor.process()


if __name__ == "__main__":
    # decorating here to allow easier tests
    main = hydra_runner(config_name="config")(main)
    main()
