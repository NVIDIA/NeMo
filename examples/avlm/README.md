# Half-duplex AVLM Dataloader

## Follow the following steps for loading the LLaVA pretraining dataset

- **Soft Link the LLaVA Dataset to Your Working Directory**
  ```
  ln -s /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/LLaVA-CC3M-Pretrain-595K/ LLaVA-CC3M-Pretrain-595K
  ```

- **Convert to Webdataset Format**
  ```
  python convert_llava-ptr_to_wds.py
  ```
- **Generate Metadata for Megatron-Energon**

  - ***Energon prepare***
  ```
  energon prepare --split-parts="train:wds/*.tar" wds
  ```
    Select no when asked for dataset.yaml preparation

  - ***Copy sample_loaders.py and dataset.yaml to .nv-meta***


- **Run the example script**
  ```
  python examples/avlm/avlm_pretrain.py --data_path path/to/tar/files
  ```
