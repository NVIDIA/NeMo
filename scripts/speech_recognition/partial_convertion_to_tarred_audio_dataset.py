from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
import json
import os
from joblib import Parallel, delayed
from omegaconf import MISSING
from typing import Optional

from convert_to_tarred_audio_dataset import ASRTarredDatasetMetadata, ASRTarredDatasetBuilder

def select_shards(manifest_filepath: str, shards_to_tar: str):
    shard_ids = []
    if ":" not in shards_to_tar:
        shard_ids = [int(shards_to_tar)]
    else:
        start_shard_idx, end_shard_idx = map(lambda x: int(x.strip()) if x.strip() else None, shards_to_tar.split(":"))
        shard_ids = list(range(start_shard_idx, end_shard_idx))
    
    entries_to_shard = {}
    with open(manifest_filepath, 'r') as manifest:
        line = manifest.readline()
        while line:
            entry = json.loads(line)
            if entry['shard_id'] in shard_ids:
                if entry['shard_id'] not in entries_to_shard:
                    entries_to_shard[entry['shard_id']] = []
                entries_to_shard[entry['shard_id']].append(entry)
            line = manifest.readline()
    
    return entries_to_shard

@dataclass
class PartialASRTarredDatasetConfig:
    tarred_manifest_filepath: str = MISSING
    output_dir: Optional[str] = None
    shards_to_tar: Optional[str] = ":"
    num_workers: int = 1
    dataset_metadata_filepath: Optional[str] = None
    dataset_metadata: ASRTarredDatasetMetadata = field(default=ASRTarredDatasetMetadata)

def create_shards(cfg: PartialASRTarredDatasetConfig):
    if cfg.tarred_manifest_filepath is None:
        raise ValueError("")

    if not os.path.exists(cfg.tarred_manifest_filepath):
        raise FileNotFoundError("")

    if cfg.dataset_metadata_filepath is None:
        cfg.dataset_metadata_filepath = os.path.join(os.path.dirname(cfg.tarred_manifest_filepath), "metadata.yaml")

    if cfg.output_dir is None:
        cfg.output_dir = os.path.dirname(cfg.tarred_manifest_filepath)
        
    if not os.path.exists(cfg.dataset_metadata_filepath):
        raise FileNotFoundError("")
    else:
        cfg.dataset_metadata = ASRTarredDatasetMetadata.from_file(cfg.dataset_metadata_filepath)
    
    entries_to_shard = select_shards(cfg.tarred_manifest_filepath, cfg.shards_to_tar)

    builder = ASRTarredDatasetBuilder()
    builder.configure(cfg.dataset_metadata.dataset_config)

    with Parallel(n_jobs=cfg.num_workers, verbose=len(entries_to_shard)) as parallel:
        # Call parallel tarfile construction
        _ = parallel(
            delayed(builder._create_shard)(entries = entries_to_shard[shard_id], 
                                            target_dir = cfg.output_dir, 
                                            shard_id = shard_id, 
                                            )
            for shard_id in entries_to_shard)


@hydra.main(config_path=None, config_name='partial_tar_config')
def main(cfg: PartialASRTarredDatasetConfig):
    create_shards(cfg)

ConfigStore.instance().store(name='partial_tar_config', node=PartialASRTarredDatasetConfig)

if __name__ == '__main__':
    main()