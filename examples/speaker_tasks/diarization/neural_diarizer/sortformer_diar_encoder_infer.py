# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import seaborn as sns
import numpy as np

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
seed_everything(42)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from nemo.collections.asr.data.audio_to_msdd_mock_label import generate_mock_embs

def plot_enc_tsne(x, targets, memo):
    # x = enc_states_list[-1].squeeze(0).cpu().detach().numpy() 
    tsne = TSNE(n_components=2, verbose=False, random_state=100)
    zembs = tsne.fit_transform(x) 

    # Step 1: Create a new column filled with 0.5
    new_column = torch.full((targets.size(0), 1), 0.5)
    # Step 2: Concatenate the new column with the original tensor
    updated_targets = torch.cat((new_column, targets), dim=1)
    
    df = pd.DataFrame()
    df["y"] = updated_targets.argmax(dim=1).detach().cpu().numpy()
    df["comp-1"] = zembs[:,0]
    df["comp-2"] = zembs[:,1]

    # Plotting using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="SortFormer HiddenState T-SNE projection")

    # Save the plot as a PNG file in the specified directory
    plt.savefig(f'/home/taejinp/Downloads/tsne_plots/tsne_sortformer_plot_{memo}.png')
    
def remove_speaker_models(ckpt_path):
    ckpt_instance = torch.load(ckpt_path)
    _state_dict = ckpt_instance['state_dict']

    key_list = list(_state_dict.keys())
    for key in key_list:
        if '_speaker_model.' in key or '_speaker_model_decoder.' in key:
            # import ipdb; ipdb.set_trace()
            del _state_dict[key]
    
    target_path = ckpt_path.replace('.ckpt', '.removed.ckpt')  
    torch.save(ckpt_instance, target_path)
    return target_path


# @hydra_runner(config_path="../conf/neural_diarizer", config_name="msdd_5scl_15_05_50Povl_256x3x32x2.yaml")
def main():
    # logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    # trainer = pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    # ckpt_path = "/disk_c/taejinp_backup/msdd_model_train/NVB_SFmr_MixMockEmbsTest/version_18_f0:84/checkpoints/e613.ckpt" 
    ckpt_path = "/disk_c/taejinp_backup/msdd_model_train/SFmr_MixMockEmbsTest/version_21/checkpoints/ep2255.ckpt"
    target_path = remove_speaker_models(ckpt_path)
    sortformer_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=target_path)
    unit_len = 25
    targets = torch.eye(4,4).repeat_interleave(unit_len,1).t()
    targets[:,2:] = 0
    # targets[:,3:] = 0
    targets = targets[:2*unit_len, :]
    new_column = torch.full((targets.size(0), 1), 0.5)
    updated_targets = torch.cat((new_column, targets), dim=1)
    mock_embs, audio_signal_length, targets = generate_mock_embs(targets=targets, seed=315, 
                                                                  mock_emb_noise_std=0.03,
                                                                  mock_emb_degree_of_freedom=4,
                                                                  min_noise_std=0.01,)
    mock_embs = mock_embs.unsqueeze(0)
    audio_signal = mock_embs
    
    audio_signal, audio_signal_length, targets  

    audio_signal = audio_signal.cuda()
    ms_seg_counts = torch.tensor([]).cuda()
    ms_seg_timestamps = torch.tensor([]).cuda()
    scale_mapping = torch.tensor([]).cuda()
    sortformer_model.alpha = 0.0
    
    _preds_mean, preds_, attn_score_stack, enc_states_list, preds_list = sortformer_model.forward(
        audio_signal=audio_signal,
        audio_signal_length=audio_signal_length,
        ms_seg_timestamps=ms_seg_timestamps,
        ms_seg_counts=ms_seg_counts,
        scale_mapping=scale_mapping,
        temp_targets=targets,
    )
    
    audio_signal_np = audio_signal.squeeze(0).cpu().detach().numpy() 
    plot_enc_tsne(audio_signal_np, targets, memo=f'input', )
    for layer_c in range(len(enc_states_list)):
        print(f"Plotting TSNE for layer {layer_c} ...")
        x = enc_states_list[layer_c].squeeze(0).cpu().detach().numpy() 
        plot_enc_tsne(x, targets, memo=f'layer{layer_c}', )
    preds = preds_.squeeze(0).cpu().detach().numpy() 
    plot_enc_tsne(preds, targets, memo=f'preds', )
    _preds_mean = _preds_mean.squeeze(0).cpu().detach().numpy() 
    plot_enc_tsne(_preds_mean, targets, memo=f'preds_mean', )
    
    # Optionally, you can also show the plot if desired
    plt.show()
    import ipdb; ipdb.set_trace()
    
    # msdd_model = SortformerEncLabelModel(cfg=cfg.model, trainer=trainer)
    # trainer.fit(msdd_model)


if __name__ == '__main__':
    main()
