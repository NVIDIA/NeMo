from dataclasses import dataclass, field
import hydra
import json
import os
from typing import List
from joblib import Parallel, delayed

from convert_to_tarred_audio_dataset import ASRTarredDatasetBuilder

def select_shards(manifest_filepath: str, shards_to_tar: str):
    shard_ids = []
    if ":" not in shards_to_tar:
        shard_ids = [int(shards_to_tar)]
    else:
        shard_ids = slice(*map(lambda x: int(x.strip()) if x.strip() else None, shards_to_tar.split(":")))

    entries_to_shard = {}
    with open(manifest_filepath, 'r') as manifest:
        line = manifest.readlines()
        while line:
            entry = json.loads(line)
            if entry['shard_id'] in shard_ids:
                if entry['shard_id'] not in entries_to_shard:
                    entries_to_shard[entry['shard_id']] = []
                entries_to_shard[entry['shard_id']].append(entry)
            line = manifest.readlines()
    
    return entries_to_shard


@dataclass
class PartialASRTarredDatasetBuilder:
    tarred_manifest_filepath: str = None
    output_dir: str = None
    shards_to_tar: List[int] = field(default_factory=list)
    num_workers: int = None


@hydra.main(config_path=None, config_name='partial_tar_config')
def main(cfg: PartialASRTarredDatasetBuilder):
    if cfg.tarred_manifest_filepath is None:
        raise ValueError("")
    
    if cfg.output_dir is None:
        cfg.output_dir = os.path.dirname(cfg.tarred_manifest_filepath)
    
    entries_to_shard = select_shards(cfg.tarred_manifest_filepath, cfg.shards_to_tar)

    builder = ASRTarredDatasetBuilder()

    with Parallel(n_jobs=cfg.num_workers, verbose=len(entries_to_shard)) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(builder._create_shard)(entries_to_shard[shard_id], cfg.output_dir, shard_id, manifest_folder)
                for shard_id in entries_to_shard)

    