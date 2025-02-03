1. Both `state_coyo_700m` and `state_cc3m` had the following parameters:
   1. Single node, Single GPU, 16 workers
   2. Batch size: 500
2. Some debug logs from coyo_700m:
```angular2html
[NeMo I 2025-01-21 16:16:02 base:158] Worker config: WorkerConfig(rank=0, world_size=1, num_workers=16, data_parallel_group=<torch.distributed.distributed_c10d.ProcessGroup object at 0x7ffd217ed9f0>, seed_offset=0, worker_debug_path=None, worker_log_level=0, _worker_debug_file=None, _worker_debug_file_worker_id=None)
[NeMo I 2025-01-21 16:16:02 base:159] Trying to get the dataset with following parameters:Path: /lustre/fsw/coreai_dlalgo_genai/datasets/coyo-700m, batch_size: 500, shuffle_buffer_size: None, max_samples_per_sequence: None,
rank=0, worker=0: shard_range=[part_00000/00000-00008_000002.tar[0, 1000), part_00000/00000-00008_000003.tar[0, 1000), part_00000/00000-00008_000004.tar[0, 1000), ...<627089>, part_00007/00520-00528_000058.tar[0, 1000), part_00007/00520-00528_000059.tar[0, 1000), part_00007/00520-00528_000060.tar[0, 611)] sum(count)=39186607
rank=0, worker=1: shard_range=[part_00007/00520-00528_000060.tar[611, 1000), part_00007/00520-00528_000061.tar[0, 1000), part_00007/00520-00528_000062.tar[0, 1000), ...<627089>, part_00015/00472-00480_000056.tar[0, 1000), part_00015/00472-00480_000057.tar[0, 1000), part_00015/00472-00480_000058.tar[0, 992)] sum(count)=39186607
rank=0, worker=2: shard_range=[part_00015/00472-00480_000058.tar[992, 1000), part_00015/00472-00480_000059.tar[0, 1000), part_00015/00472-00480_000060.tar[0, 1000), ...<627089>, part_00023/00416-00424_000062.tar[0, 1000), part_00023/00416-00424_000063.tar[0, 1000), part_00023/00416-00424_000064.tar[0, 337)] sum(count)=39186607
rank=0, worker=3: shard_range=[part_00023/00416-00424_000064.tar[337, 1000), part_00023/00416-00424_000065.tar[0, 1000), part_00023/00416-00424_000066.tar[0, 1000), ...<627089>, part_00031/00368-00376_000010.tar[0, 1000), part_00031/00368-00376_000011.tar[0, 1000), part_00031/00368-00376_000012.tar[0, 572)] sum(count)=39186607
rank=0, worker=4: shard_range=[part_00031/00368-00376_000012.tar[572, 1000), part_00031/00368-00376_000013.tar[0, 1000), part_00031/00368-00376_000014.tar[0, 1000), ...<627089>, part_00039/00312-00320_000020.tar[0, 1000), part_00039/00312-00320_000021.tar[0, 1000), part_00039/00312-00320_000022.tar[0, 3)] sum(count)=39186607
rank=0, worker=5: shard_range=[part_00039/00312-00320_000022.tar[3, 1000), part_00039/00312-00320_000023.tar[0, 1000), part_00039/00312-00320_000024.tar[0, 1000), ...<627089>, part_00047/00256-00264_000019.tar[0, 1000), part_00047/00256-00264_000020.tar[0, 1000), part_00047/00256-00264_000021.tar[0, 649)] sum(count)=39186607
rank=0, worker=6: shard_range=[part_00047/00256-00264_000021.tar[649, 1000), part_00047/00256-00264_000022.tar[0, 1000), part_00047/00256-00264_000023.tar[0, 1000), ...<627089>, part_00055/00200-00208_000031.tar[0, 1000), part_00055/00200-00208_000032.tar[0, 1000), part_00055/00200-00208_000033.tar[0, 500)] sum(count)=39186607
rank=0, worker=7: shard_range=[part_00055/00200-00208_000033.tar[500, 1000), part_00055/00200-00208_000034.tar[0, 1000), part_00055/00200-00208_000035.tar[0, 1000), ...<627089>, part_00063/00152-00160_000005.tar[0, 1000), part_00063/00152-00160_000006.tar[0, 1000), part_00063/00152-00160_000007.tar[0, 264)] sum(count)=39186608
rank=0, worker=8: shard_range=[part_00063/00152-00160_000007.tar[264, 1000), part_00063/00152-00160_000008.tar[0, 1000), part_00063/00152-00160_000009.tar[0, 1000), ...<627089>, part_00071/00096-00104_000009.tar[0, 1000), part_00071/00096-00104_000010.tar[0, 1000), part_00071/00096-00104_000011.tar[0, 628)] sum(count)=39186607
rank=0, worker=9: shard_range=[part_00071/00096-00104_000011.tar[628, 1000), part_00071/00096-00104_000012.tar[0, 1000), part_00071/00096-00104_000013.tar[0, 1000), ...<627089>, part_00079/00040-00048_000036.tar[0, 1000), part_00079/00040-00048_000037.tar[0, 1000), part_00079/00040-00048_000038.tar[0, 158)] sum(count)=39186607
rank=0, worker=10: shard_range=[part_00079/00040-00048_000038.tar[158, 1000), part_00079/00040-00048_000039.tar[0, 1000), part_00079/00040-00048_000040.tar[0, 1000), ...<627089>, part_00086/00568-00576_000056.tar[0, 1000), part_00086/00568-00576_000057.tar[0, 1000), part_00086/00568-00576_000058.tar[0, 44)] sum(count)=39186607
rank=0, worker=11: shard_range=[part_00086/00568-00576_000058.tar[44, 1000), part_00086/00568-00576_000059.tar[0, 1000), part_00086/00568-00576_000060.tar[0, 1000), ...<627089>, part_00096/00216-00224_000027.tar[0, 1000), part_00096/00216-00224_000028.tar[0, 1000), part_00096/00216-00224_000029.tar[0, 77)] sum(count)=39186607
rank=0, worker=12: shard_range=[part_00096/00216-00224_000029.tar[77, 1000), part_00096/00216-00224_000030.tar[0, 1000), part_00096/00216-00224_000031.tar[0, 1000), ...<627089>, part_00104/00160-00168_000021.tar[0, 1000), part_00104/00160-00168_000022.tar[0, 1000), part_00104/00160-00168_000023.tar[0, 5)] sum(count)=39186607
rank=0, worker=13: shard_range=[part_00104/00160-00168_000023.tar[5, 1000), part_00104/00160-00168_000024.tar[0, 1000), part_00104/00160-00168_000025.tar[0, 1000), ...<627089>, part_00112/00104-00112_000038.tar[0, 1000), part_00112/00104-00112_000039.tar[0, 1000), part_00112/00104-00112_000040.tar[0, 661)] sum(count)=39186607
rank=0, worker=14: shard_range=[part_00112/00104-00112_000040.tar[661, 1000), part_00112/00104-00112_000041.tar[0, 1000), part_00112/00104-00112_000042.tar[0, 1000), ...<627089>, part_00120/00048-00056_000049.tar[0, 1000), part_00120/00048-00056_000050.tar[0, 1000), part_00120/00048-00056_000051.tar[0, 70)] sum(count)=39186607
rank=0, worker=15: shard_range=[part_00120/00048-00056_000051.tar[70, 1000), part_00120/00048-00056_000052.tar[0, 1000), part_00120/00048-00056_000053.tar[0, 1000), ...<627089>, part_00127/00576-00584_000062.tar[0, 1000), part_00127/00576-00584_000063.tar[0, 1000), part_00127/00576-00584_000064.tar[0, 1000)] sum(count)=39186608
[NeMo I 2025-01-21 16:18:23 base:176] Loaded the Dataset




[NeMo I 2025-01-21 16:19:57 base:294] Multimodal state saved in 25.383262634277344 seconds
[NeMo I 2025-01-21 16:19:57 base:298] Multimodal data loader saving dataloader state dict consumed samples 5000
[NeMo W 2025-01-21 16:20:23 nemo_logging:361] /usr/local/lib/python3.10/dist-packages/pyannote/core/notebook.py:134: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
      cm = get_cmap("Set1")

[NeMo I 2025-01-21 16:20:45 base:294] Multimodal state saved in 18.832727670669556 seconds
[NeMo I 2025-01-21 16:20:45 base:298] Multimodal data loader saving dataloader state dict consumed samples 5000
```
3. Some Debug logs from cc3m:
```angular2html
[NeMo I 2025-01-21 16:30:25 base:158] Worker config: WorkerConfig(rank=0, world_size=1, num_workers=16, data_parallel_group=<torch.distributed.distributed_c10d.ProcessGroup object at 0x7ffd1bbaabf0>, seed_offset=0, worker_debug_path=None, worker_log_level=0, _worker_debug_file=None, _worker_debug_file_worker_id=None)
[NeMo I 2025-01-21 16:30:25 base:159] Trying to get the dataset with following parameters:Path: /lustre/fsw/coreai_dlalgo_genai/datasets/cc3m_abhi, batch_size: 500, shuffle_buffer_size: None, max_samples_per_sequence: None,
rank=0, worker=0: shard_range=[00000-00008_000000.tar[0, 1000), 00000-00008_000001.tar[0, 1000), 00000-00008_000002.tar[0, 1000), ...<703>, 00016-00024_000004.tar[0, 1000), 00016-00024_000005.tar[0, 1000), 00016-00024_000006.tar[0, 312)] sum(count)=44312
rank=0, worker=1: shard_range=[00016-00024_000006.tar[312, 1000), 00016-00024_000007.tar[0, 1000), 00016-00024_000008.tar[0, 1000), ...<703>, 00032-00040_000010.tar[0, 1000), 00032-00040_000011.tar[0, 1000), 00032-00040_000012.tar[0, 625)] sum(count)=44313
rank=0, worker=2: shard_range=[00032-00040_000012.tar[625, 1000), 00032-00040_000013.tar[0, 1000), 00032-00040_000014.tar[0, 1000), ...<703>, 00048-00056_000016.tar[0, 1000), 00048-00056_000017.tar[0, 1000), 00048-00056_000018.tar[0, 937)] sum(count)=44312
rank=0, worker=3: shard_range=[00048-00056_000018.tar[937, 1000), 00056-00064_000000.tar[0, 1000), 00056-00064_000001.tar[0, 1000), ...<703>, 00072-00080_000004.tar[0, 1000), 00072-00080_000005.tar[0, 1000), 00072-00080_000006.tar[0, 250)] sum(count)=44313
rank=0, worker=4: shard_range=[00072-00080_000006.tar[250, 1000), 00072-00080_000007.tar[0, 1000), 00072-00080_000008.tar[0, 1000), ...<703>, 00088-00096_000010.tar[0, 1000), 00088-00096_000011.tar[0, 1000), 00088-00096_000012.tar[0, 562)] sum(count)=44312
rank=0, worker=5: shard_range=[00088-00096_000012.tar[562, 1000), 00088-00096_000013.tar[0, 1000), 00088-00096_000014.tar[0, 1000), ...<703>, 00104-00112_000016.tar[0, 1000), 00104-00112_000017.tar[0, 1000), 00104-00112_000018.tar[0, 875)] sum(count)=44313
rank=0, worker=6: shard_range=[00104-00112_000018.tar[875, 1000), 00112-00120_000000.tar[0, 1000), 00112-00120_000001.tar[0, 1000), ...<703>, 00128-00136_000004.tar[0, 1000), 00128-00136_000005.tar[0, 1000), 00128-00136_000006.tar[0, 187)] sum(count)=44312
rank=0, worker=7: shard_range=[00128-00136_000006.tar[187, 1000), 00128-00136_000007.tar[0, 1000), 00128-00136_000008.tar[0, 1000), ...<703>, 00144-00152_000010.tar[0, 1000), 00144-00152_000011.tar[0, 1000), 00144-00152_000012.tar[0, 500)] sum(count)=44313
rank=0, worker=8: shard_range=[00144-00152_000012.tar[500, 1000), 00144-00152_000013.tar[0, 1000), 00144-00152_000014.tar[0, 1000), ...<703>, 00160-00168_000016.tar[0, 1000), 00160-00168_000017.tar[0, 1000), 00160-00168_000018.tar[0, 812)] sum(count)=44312
rank=0, worker=9: shard_range=[00160-00168_000018.tar[812, 1000), 00168-00176_000000.tar[0, 1000), 00168-00176_000001.tar[0, 1000), ...<703>, 00184-00192_000004.tar[0, 1000), 00184-00192_000005.tar[0, 1000), 00184-00192_000006.tar[0, 125)] sum(count)=44313
rank=0, worker=10: shard_range=[00184-00192_000006.tar[125, 1000), 00184-00192_000007.tar[0, 1000), 00184-00192_000008.tar[0, 1000), ...<703>, 00200-00208_000010.tar[0, 1000), 00200-00208_000011.tar[0, 1000), 00200-00208_000012.tar[0, 437)] sum(count)=44312
rank=0, worker=11: shard_range=[00200-00208_000012.tar[437, 1000), 00200-00208_000013.tar[0, 1000), 00200-00208_000014.tar[0, 1000), ...<703>, 00216-00224_000016.tar[0, 1000), 00216-00224_000017.tar[0, 1000), 00216-00224_000018.tar[0, 750)] sum(count)=44313
rank=0, worker=12: shard_range=[00216-00224_000018.tar[750, 1000), 00224-00232_000000.tar[0, 1000), 00224-00232_000001.tar[0, 1000), ...<703>, 00240-00248_000004.tar[0, 1000), 00240-00248_000005.tar[0, 1000), 00240-00248_000006.tar[0, 62)] sum(count)=44312
rank=0, worker=13: shard_range=[00240-00248_000006.tar[62, 1000), 00240-00248_000007.tar[0, 1000), 00240-00248_000008.tar[0, 1000), ...<703>, 00256-00264_000010.tar[0, 1000), 00256-00264_000011.tar[0, 1000), 00256-00264_000012.tar[0, 375)] sum(count)=44313
rank=0, worker=14: shard_range=[00256-00264_000012.tar[375, 1000), 00256-00264_000013.tar[0, 1000), 00256-00264_000014.tar[0, 1000), ...<703>, 00272-00280_000016.tar[0, 1000), 00272-00280_000017.tar[0, 1000), 00272-00280_000018.tar[0, 687)] sum(count)=44312
rank=0, worker=15: shard_range=[00272-00280_000018.tar[687, 1000), 00280-00288_000000.tar[0, 1000), 00280-00288_000001.tar[0, 1000), ...<703>, 00296-00304_000003.tar[0, 1000), 00296-00304_000004.tar[0, 1000), 00296-00304_000005.tar[0, 1000)] sum(count)=44313
[NeMo I 2025-01-21 16:30:25 base:176] Loaded the Dataset




[NeMo I 2025-01-21 16:22:32 base:294] Multimodal state saved in 5.108197450637817 seconds
[NeMo I 2025-01-21 16:22:32 base:298] Multimodal data loader saving dataloader state dict consumed samples 5000
[NeMo W 2025-01-21 16:22:39 nemo_logging:361] /usr/local/lib/python3.10/dist-packages/pyannote/core/notebook.py:134: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
      cm = get_cmap("Set1")

[NeMo I 2025-01-21 16:22:42 base:294] Multimodal state saved in 0.01719832420349121 seconds
[NeMo I 2025-01-21 16:22:42 base:298] Multimodal data loader saving dataloader state dict consumed samples 5000
```

4. The code used to save the state was `save_state()`: https://github.com/NVIDIA/NeMo/blob/wip_ab/clip/nemo/collections/multimodal/data/energon/base.py#L273

# Questions that we had:
1. Why are the sizes of the data loader states so different?
2. Why is the saving state of one dataloader taking much longer than the other one?