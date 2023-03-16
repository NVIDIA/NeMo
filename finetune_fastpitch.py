"""finetune a Fastpitch model on a new dataset

example usage: TODO
"""

import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from tqdm import tqdm
import json
import subprocess


def get_pitch_stats(dataset_path, sup_data_path):
    # hydra has such a complicated codebase I'm going to create the config manually here
    config_path = 'scripts/dataset_processing/tts/ljspeech/ds_conf/ds_for_fastpitch_align.yaml'

    variables_used_in_config = [
        'whitelist_path',
        'phoneme_dict_path',
        'heteronyms_path',
        'sup_data_types',
    ]

    with open(config_path) as f:
        raw_file = f.read()
        file_yaml = yaml.safe_load(raw_file)

        for variable_to_replace in variables_used_in_config:
            raw_file = raw_file.replace(
                f'${{{variable_to_replace}}}',
                str(file_yaml[variable_to_replace])
            )

    config_yaml = yaml.safe_load(raw_file)
    config = OmegaConf.create(config_yaml)
    config.manifest_filepath = dataset_path
    config.dataset.manifest_filepath = dataset_path
    config.sup_data_path = sup_data_path
    config.dataset.sup_data_path = sup_data_path

    dataset = instantiate(config.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=dataset._collate_fn,
        num_workers=0,
    )
    sample_rate = dataset.sample_rate
    pitch_list = []
    durations = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        audios, audio_lengths, tokens, tokens_lengths, align_prior_matrices, pitches, pitches_lengths = batch
        pitch = pitches.squeeze(0)
        pitch_list += [pitch[pitch != 0]]
        durations += [audio_lengths.squeeze(0) / sample_rate]

    pitch_tensor = torch.cat(pitch_list)
    pitch_mean, pitch_std = pitch_tensor.mean().item(), pitch_tensor.std().item()
    pitch_min, pitch_max = pitch_tensor.min().item(), pitch_tensor.max().item()

    return pitch_mean, pitch_std, pitch_min, pitch_max, sum(durations)


if __name__ == '__main__':
    # train_dataset_path = 'data/nichole/data_more.json'
    # validation_dataset_path = 'data/nichole/data.json'
    # sup_data_path = 'data/nichole/train_data_cache'
    # dest = './nichole_model'
    train_dataset_path = 'data/nichole/train_manifest_0316.json'
    validation_dataset_path = 'data/nichole/data.json'
    sup_data_path = 'data/nichole/train_data_cache'
    dest = './nichole_model'
    # train_dataset_path = 'NickyData 2/clean_train_manifest.json'
    # # validation_dataset_path = 'data/NickyData/validation_manifest.json'
    # sup_data_path = 'data/NickyData 2/something'
    # dest = './nicky_model'

    pitch_mean, pitch_std, pitch_fmin, pitch_fmax, duration = get_pitch_stats(
        train_dataset_path, sup_data_path)

    print(pitch_mean, pitch_std, pitch_fmin, pitch_fmax)
    print(duration / 60)
    # 1000 - steps per minute of audio
    max_steps = (duration / 60) * 1000

    args = [
        'python',
        'examples/tts/fastpitch_finetune.py',
        '--config-name=fastpitch_align_v1.05.yaml',
        f'train_dataset={train_dataset_path}',
        f'validation_datasets={validation_dataset_path}',
        f'sup_data_path={sup_data_path}',
        f'exp_manager.exp_dir={dest}',
        '+init_from_pretrained_model=tts_en_fastpitch',
        f'+trainer.max_steps={max_steps}',
        '~trainer.max_epochs',
        'trainer.check_val_every_n_epoch=25',
        'model.train_ds.dataloader_params.batch_size=16',
        'model.validation_ds.dataloader_params.batch_size=16',
        'model.n_speakers=1',
        f'model.pitch_mean={pitch_mean}',
        f'model.pitch_std={pitch_std}',
        f'model.pitch_fmin={pitch_fmin}',
        f'model.pitch_fmax={pitch_fmax}',
        'model.optim.lr=2e-4',
        '~model.optim.sched',
        'model.optim.name=adam',
        'trainer.devices=1',
        'trainer.strategy=null',
        '+model.text_tokenizer.add_blank_at=true',
    ]
    newline = '\n'
    print(f'Funning finetune command with: {newline.join(args)}')
    subprocess.run(' '.join(args), shell=True)
