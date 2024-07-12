# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import NevaWordEmbeddingMixin


class LitaWordEmbeddingMixin(NevaWordEmbeddingMixin):
    def init(self, mm_cfg):
        lita_conf = mm_cfg.get('lita', {})
        lita_video_arch=lita_conf.get('lita_video_arch', 'temporal_spatial_pool')
        visual_token_format=lita_conf.get('visual_token_format', 'v1')
        use_media_start_end=mm_cfg.get('use_im_start_end', False)  # we need to make this clear
        sample_frames=lita_conf.get('sample_frames', 4)
        self.init_lita(
            lita_video_arch=lita_video_arch,
            visual_token_format=visual_token_format,
            use_media_start_end=use_media_start_end,
            sample_frames=sample_frames,
        )

    def init_lita(
        self,
        lita_video_arch: str,
        visual_token_format: str = "v1",
        use_media_start_end: bool = False,
        sample_frames: int = 4,
    ):
        """_summary_

        Args:
            lita_video_arch (str): ['temporal_spatial_pool', 'temporal_spatial', 'temporal_all_resolution']
            visual_token_format (str, optional): default to 'v1', other option ["v1", "im_vid_start_end"]
                v1: no video_start_id and video_end_id, video tokens are inserted between fast/slow (temporal/spatial) tokens
                im_vid_start_end: video start and end tokens are inserted before and after temporal tokens
                                  image start and end tokens are inserted before and after spatial tokens
            use_media_start_end (bool, optional):
                whether media start and media end is used in input_ids, Defaults to False.
                Notice, when it is false, the media_start_id and media_end_id will play as an placeholder
                input_ids = [..., media_start_id, t1, t2, t3...., media_end_id, ...]
                use_media_start_end = False
                    we will replace the tokens including and between: [media_start_id, ... media_end_id]
                use_media_start_end = True
                    we will replace the tokens between: (media_start_id, ... media_end_id)
            num_frames (int, optional): number of frames to sample from the video, default to 4
        """
        self.lita_video_arch = lita_video_arch
        self.visual_token_format = visual_token_format
        self.use_media_start_end = use_media_start_end
        self.sample_frames = sample_frames

    def add_lita_layer(self, media_features):
        """_summary_

        Args:
            media_features (torch.Tensor):
                feature after encoded by vision encoder
                shape: Batch, T (number of images), S (num patches), H (hidden  size)
        Returns:
            tokens (torch.Tensor):
                shape: Batch, T + M, D (hidden size)
        """

        b, T, S, H = media_features.shape
        tokens = media_features
        if self.lita_video_arch == 'temporal_spatial_pool':
            pool_size = 2
            h = w = int(np.sqrt(S))
            selected_frames = np.round(np.linspace(0, tokens.shape[1] - 1, pool_size * pool_size)).astype(int)
            s_tokens = tokens[:, selected_frames, ...]
            s_tokens = rearrange(s_tokens, 'b t (h w) d -> (b t) d h w', h=h, w=w)
            s_tokens = F.avg_pool2d(s_tokens, kernel_size=pool_size)
            s_tokens = rearrange(s_tokens, '(b t) d h w -> b (t h w) d', b=b)  # B, M, D
            t_tokens = reduce(tokens, 'b t s d -> b t d', 'mean')
            # tokens = torch.cat([t_tokens, s_tokens], dim=1)  # B, T + M, D
            return t_tokens, s_tokens
        elif self.lita_video_arch == 'temporal_spatial':
            t_tokens = reduce(tokens, 'b t s d -> b t d', 'mean')
            s_tokens = reduce(tokens, 'b t s d -> b s d', 'mean')
            # tokens = torch.cat([t_tokens, s_tokens], dim=1)  # B, T + M, D
            return t_tokens, s_tokens
        elif self.lita_video_arch == 'temporal_all_resolution':
            idx = np.round(np.linspace(0, tokens.shape[1] - 1, self.sample_frames)).astype(int)
            im_features = tokens[:, idx, ...]  # B, num_frames, S, D
            # im_tokens = im_features.view(b, -1, H) # flatten the B, num_frames * S, D
            im_tokens = im_features
            vid_tokens = reduce(tokens, 'b t s d -> b t d', 'mean')
            # s and t tokens have been changed position
            return im_tokens, vid_tokens
        else:
            raise ValueError(f"Unknown video architecture: {self.lita_video_arch}")

    def replace_media_embeddings(self, input_ids, inputs_embeds, media):
        """_summary_

        Args:
            input_ids (torch.tensor): The input token ids [B, T]
            words_embeddings (torch.tensor): The input embeddings [B, T, D]
            media (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
        """
        if input_ids.shape[1] == 1:
            return inputs_embeds

        if media is None:
            return inputs_embeds
        if type(media) is list:
            raise NotImplementedError("dynamic length of videos not supported yet, only fixed length of videos now")
        # 1, 1, num_frames, 3, 244, 244
        media_features = self.encode_vision_x(media)  # B T F S(eq) H(idden)
        B, T, F, S, H = media_features.shape
        assert T == 1, "multiple videos per sample not supported yet"
        media_features = media_features.squeeze(1)
        t_tokens, s_tokens = self.add_lita_layer(media_features)  # B, T, D & B, M, D
        T = t_tokens.shape[1]
        M = s_tokens.shape[1]
        inputs_embeds = inputs_embeds.clone()
        for idx, input_id in enumerate(input_ids):
            media_start_position = torch.where(input_id == self.media_start_id)[0]
            media_end_position = torch.where(input_id == self.media_end_id)[0]
            if self.visual_token_format != 'im_vid_start_end':
                assert len(media_start_position) == 1, "Only 1 video per sample supported"
                assert len(media_end_position) == 1, "Only 1 video per sample supported"

            media_start_position = media_start_position[0]
            media_end_position = media_end_position[-1]
            if self.use_media_start_end:
                # replace the tokens between media_start_id and media_end_id
                start, end = media_start_position + 1, media_end_position - 1
            else:
                # replace the tokens including and between media_start_id and media_end_id
                start, end = media_start_position, media_end_position

            if self.visual_token_format == 'v1':
                t_token_start, t_token_end = start, start + T
                s_token_start, s_token_end = start + T, start + T + M
                assert s_token_end == end + 1, "Token replacement error"
                inputs_embeds[idx, t_token_start:t_token_end] = t_tokens[idx]
                inputs_embeds[idx, s_token_start:s_token_end] = s_tokens[idx]
            elif self.visual_token_format == 'im_vid_start_end':  # v1.5 lita
                if not self.use_media_start_end:
                    # replace the media start and media end embedding with
                    # img_start and vid_end token embedding
                    inputs_embeds[idx, start] = inputs_embeds[idx, start + 1]
                    inputs_embeds[idx, end] = inputs_embeds[idx, end - 1]
                # TO DO: To optimize the below codes
                im_features, vid_features = t_tokens[idx], s_tokens[idx]
                # im_feature: num_frames * S, D
                emb_start = start + 1  # skip the img_start token
                num_frames, S, D = im_features.shape
                for i in range(num_frames):
                    inputs_embeds[idx, emb_start : emb_start + S] = im_features[i]
                    emb_start = emb_start + S + 2  # skip the img_end token and img_start token
                T = vid_features.shape[0]
                inputs_embeds[idx, emb_start : emb_start + T] = vid_features
                assert emb_start + T == end
            else:
                raise ValueError(f"Unsupported visual_token_format {self.visual_token_format}")
        return inputs_embeds
