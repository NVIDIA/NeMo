import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


def run_asr_inference(cfg: DictConfig) -> DictConfig:
    if (cfg.model_path and cfg.pretrained_name) or (not cfg.model_path and not cfg.pretrained_name):
        raise ValueError("Please specify either cfg.model_path or cfg.pretrained_name!")

    if cfg.inference_mode.mode == "offline":
        # Specify default total_buffer_in_secs=22 and chunk_len_in_secs=20 for offline conformer
        # to avoid problem of long audio sample.
        OmegaConf.set_struct(cfg, True)
        if (cfg.model_path and 'conformer' in cfg.model_path.lower()) or (
            cfg.pretrained_name and 'conformer' in cfg.pretrained_name.lower()
        ):
            if 'total_buffer_in_secs' not in cfg.inference_mode or not cfg.inference_mode.total_buffer_in_secs:
                with open_dict(cfg):
                    cfg.inference_mode.total_buffer_in_secs = 22
            if 'chunk_len_in_secs' not in cfg.inference_mode or not cfg.inference_mode.chunk_len_in_secs:
                with open_dict(cfg):
                    cfg.inference_mode.chunk_len_in_secs = 20

            cfg = run_chunked_inference(cfg)
        else:
            cfg = run_offline_inference(cfg)

    elif cfg.inference_mode.mode == "chunked":
        if (
            "total_buffer_in_secs" not in cfg.inference_mode
            or "chunk_len_in_secs" not in cfg.inference_mode
            or not cfg.inference_mode.total_buffer_in_secs
            or not cfg.inference_mode.chunk_len_in_secs
        ):
            raise ValueError(
                f"Please specify both total_buffer_in_secs and chunk_len_in_secs for chunked inference_mode"
            )
        cfg = run_chunked_inference(cfg)

    else:
        raise ValueError(f"inference_mode could only be offline or chunked, but got {cfg.inference_mode.mode}")

    return cfg


def run_chunked_inference(cfg: DictConfig) -> DictConfig:
    if "output_filename" not in cfg or not cfg.output_filename:
        if cfg.model_path:
            model_name = Path(cfg.model_path).setup_model
        else:
            model_name = cfg.pretrained_name
        dataset_name = Path(cfg.test_ds.manifest_filepath).stem
        mode_name = (
            cfg.inference_mode.mode
            + "B"
            + str(cfg.inference_mode.total_buffer_in_secs)
            + "C"
            + str(cfg.inference_mode.chunk_len_in_secs)
        )

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.output_filename = model_name + "-" + dataset_name + "-" + mode_name + ".json"

    script_path = (
        Path(__file__).parents[5]
        / "examples"
        / "asr"
        / "asr_chunked_inference"
        / "ctc"
        / "speech_to_text_buffered_infer_ctc.py"
    )

    if (cfg.pretrained_name and 'transducer' in cfg.pretrained_name) or (
        cfg.model_path and 'transducer' in cfg.model_path
    ):
        script_path = (
            Path(__file__).parents[5]
            / "examples"
            / "asr"
            / "asr_chunked_inference"
            / "rnnt"
            / "speech_to_text_buffered_infer_rnnt.py"
        )

    subprocess.run(
        f"python {script_path} "
        f"model_path={cfg.model_path} "
        f"pretrained_name={cfg.pretrained_name} "
        f"dataset_manifest={cfg.test_ds.manifest_filepath} "
        f"output_filename={cfg.output_filename} "
        f"batch_size={cfg.test_ds.batch_size} "
        f"chunk_len_in_secs={cfg.inference_mode.chunk_len_in_secs} "
        f"total_buffer_in_secs={cfg.inference_mode.total_buffer_in_secs} "
        f"model_stride={cfg.inference_mode.model_stride} ",
        shell=True,
        check=True,
    )
    return cfg


def run_offline_inference(cfg: DictConfig) -> DictConfig:
    if "output_filename" not in cfg or not cfg.output_filename:
        if cfg.model_path:
            model_name = Path(cfg.model_path).setup_model
        else:
            model_name = cfg.pretrained_name
        dataset_name = Path(cfg.test_ds.manifest_filepath).stem
        mode_name = cfg.inference_mode.mode

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.output_filename = model_name + "-" + dataset_name + "-" + mode_name + ".json"

    script_path = Path(__file__).parents[5] / "examples" / "asr" / "transcribe_speech.py"

    subprocess.run(
        f"python {script_path} "
        f"model_path={cfg.model_path} "
        f"pretrained_name={cfg.pretrained_name} "
        f"dataset_manifest={cfg.test_ds.manifest_filepath} "
        f"output_filename={cfg.output_filename} "
        f"batch_size={cfg.test_ds.batch_size} ",
        shell=True,
        check=True,
    )

    return cfg
