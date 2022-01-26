###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import torch
from torch import nn
from nemo.collections.tts.helpers.common import Encoder, ConvNorm, LengthRegulator, PositionalEmbedding
from nemo.collections.tts.helpers.common import Invertible1x1Conv, Invertible1x1ConvL, Invertible1x1ConvLU
from nemo.collections.tts.helpers.common import Invertible1x1ConvLUS
from nemo.collections.tts.helpers.common import ConvAttention
from nemo.collections.tts.helpers.common import AffineTransformationLayer
from nemo.collections.tts.helpers.common import get_mask_from_lengths, shift_and_scale_data
from nemo.collections.tts.modules.feature_prediction_model import get_FP_model
from nemo.collections.tts.modules.alignment import mas
from nemo.collections.tts.modules.radttstransformer import FFTransformer
from nemo.utils import logging


class FlowStep(nn.Module):
    def __init__(self,  n_mel_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', use_feature_gating=False,
                 scaling_fn='exp', matrix_decomposition='',
                 affine_activation='softplus', enable_lstm=False,
                 remove_wn_conv=False, p_dropout=0.0):
        super(FlowStep, self).__init__()
        if matrix_decomposition == 'LU':
            self.invtbl_conv = Invertible1x1ConvLU(n_mel_channels)
        elif matrix_decomposition == 'L':
            self.invtbl_conv = Invertible1x1ConvL(n_mel_channels)
        elif matrix_decomposition == 'LUS':
            self.invtbl_conv = Invertible1x1ConvLUS(n_mel_channels)
        else:
            self.invtbl_conv = Invertible1x1Conv(n_mel_channels)

        self.affine_tfn = AffineTransformationLayer(
            n_mel_channels, n_context_dim, n_layers, affine_model=affine_model,
            scaling_fn=scaling_fn, use_feature_gating=use_feature_gating,
            affine_activation=affine_activation, enable_lstm=enable_lstm,
            p_dropout=p_dropout)

    def forward(self, z, context, reverse=False):
        if reverse:  # for inference z-> mel
            z = self.affine_tfn(z, context, reverse)
            z = self.invtbl_conv(z, reverse)
            return z
        else:  # training mel->z
            z, log_det_W = self.invtbl_conv(z)
            z, log_s = self.affine_tfn(z, context)
            return z, log_det_W, log_s


class RadTTSModule(torch.nn.Module):
    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_conv_layers_per_step, n_mel_channels, n_hidden,
                 mel_encoder_n_hidden, n_components, fixed_gaussian,
                 mean_scale, dummy_speaker_embedding, use_positional_embedding,
                 n_early_size, n_early_every, n_group_size, use_feature_gating,
                 affine_model, dur_model_config, feature_model_config,
                 what_to_train=['ftp'], scaling_fn='exp',
                 reduction_norm='', matrix_decomposition='',
                 learn_alignments=False, text_encoder_name='tacotron2',
                 affine_activation='softplus', align_query_enc_type='3xconv',
                 use_query_proj=True, attn_use_CTC=True,
                 lstm_applicable_steps=[], use_context_lstm=False,
                 context_lstm_norm=None, text_encoder_lstm_norm=None,
                 use_text_conditional_priors=False, n_aug_dims=0, n_f0_dims=0,
                 n_energy_avg_dims=0, context_lstm_w_f0_and_energy=True,
                 zero_out_context=False, p_dropout=0.0,
                 noise_to_pvoiced=0.0, noise_to_unvoiced_in_f0=0.0):

        super(RadTTSModule, self).__init__()
        assert(n_early_size % 2 == 0)
        self.noise_to_pvoiced = noise_to_pvoiced
        self.noise_to_unvoiced_in_f0 = noise_to_unvoiced_in_f0
        self.n_mel_channels = n_mel_channels
        self.n_f0_dims = n_f0_dims  # >= 1 to trains with f0
        self.n_energy_avg_dims = n_energy_avg_dims  # >= 1 trains with energy

        self.n_aug_dims = n_aug_dims  # last spk vec dims be used for this
        n_speaker_dim += n_aug_dims  # extend the speaker vector
        self.n_speaker_dim = n_speaker_dim
        assert(self.n_speaker_dim % 2 == 0)
        self.speaker_embedding = torch.nn.Embedding(
            n_speakers, self.n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        if text_encoder_name == 'tacotron2':
            norm_fn = nn.InstanceNorm1d
            self.encoder = Encoder(encoder_embedding_dim=n_text_dim,
                                   norm_fn=norm_fn,
                                   lstm_norm_fn=text_encoder_lstm_norm)
        elif text_encoder_name == 'transformer':
            self.encoder = FFTransformer(n_text, n_text_dim)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.learn_alignments = learn_alignments  # lrn align with attn mech
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_emb = PositionalEmbedding(n_text_dim, max_len=5000)
        self.affine_activation = affine_activation
        self.align_query_enc_type = align_query_enc_type

        self.lstm_applicable_steps = lstm_applicable_steps
        self.what_to_train = what_to_train
        self.use_query_proj = bool(use_query_proj)
        self.align_query_enc_type = align_query_enc_type
        self.attn_use_CTC = bool(attn_use_CTC)
        self.use_context_lstm = bool(use_context_lstm)
        self.context_lstm_norm = context_lstm_norm
        self.context_lstm_w_f0_and_energy = context_lstm_w_f0_and_energy
        self.use_text_conditional_priors = bool(use_text_conditional_priors)
        self.length_regulator = LengthRegulator()
        self.zero_out_context = zero_out_context
        self.p_dropout = p_dropout

        if 'dpf' in what_to_train:
            dur_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.dur_pred_layer = get_FP_model(dur_model_config)

        if 'fev-pf' in what_to_train:
            feature_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.feature_pred_layer = get_FP_model(feature_model_config)

        if 'atn' in what_to_train or 'ftp' in what_to_train:
            if self.learn_alignments:
                # defaults seem fine
                self.attention = ConvAttention(
                    n_mel_channels, self.n_speaker_dim, n_text_dim,
                    use_query_proj=self.use_query_proj,
                    align_query_enc_type=self.align_query_enc_type,
                    passthru=self.use_text_conditional_priors)

            self.n_flows = n_flows
            self.n_group_size = n_group_size

            n_flowstep_cond_dims = (
                self.n_speaker_dim +
                (n_text_dim + n_f0_dims + n_energy_avg_dims) * n_group_size)

            if self.use_context_lstm:
                n_in_context_lstm = (
                    self.n_speaker_dim + n_text_dim * n_group_size)
                n_context_lstm_hidden = int(
                    (self.n_speaker_dim + n_text_dim * n_group_size) / 2)

                if self.context_lstm_w_f0_and_energy:
                    n_in_context_lstm = (
                        n_f0_dims + n_energy_avg_dims + n_text_dim)
                    n_in_context_lstm *= n_group_size
                    n_in_context_lstm += self.n_speaker_dim

                    n_context_hidden = (
                        n_f0_dims + n_energy_avg_dims + n_text_dim)
                    n_context_hidden = n_context_hidden * n_group_size / 2
                    n_context_hidden = self.n_speaker_dim + n_context_hidden
                    n_context_hidden = int(n_context_hidden)

                    n_flowstep_cond_dims = (
                        self.n_speaker_dim + n_text_dim * n_group_size)

                self.context_lstm = torch.nn.LSTM(
                    input_size=n_in_context_lstm,
                    hidden_size=n_context_lstm_hidden, num_layers=1,
                    batch_first=True, bidirectional=True)

                if context_lstm_norm is not None:
                    if 'spectral' in context_lstm_norm:
                        logging.info("Applying spectral norm to context encoder LSTM")
                        lstm_norm_fn_pntr = torch.nn.utils.spectral_norm
                    elif 'weight' in context_lstm_norm:
                        logging.info("Applying weight norm to context encoder LSTM")
                        lstm_norm_fn_pntr = torch.nn.utils.weight_norm

                    self.context_lstm = lstm_norm_fn_pntr(
                        self.context_lstm, 'weight_hh_l0')
                    self.context_lstm = lstm_norm_fn_pntr(
                        self.context_lstm, 'weight_hh_l0_reverse')

            if self.use_text_conditional_priors:
                # NOTE: operates on UNFOLDED (aka squeezed) input
                # (output of preprocess_context)
                self.prior_predictor = nn.Sequential(
                    ConvNorm(n_text_dim, n_text_dim*2, kernel_size=3,
                             bias=True, w_init_gain='relu'),
                    torch.nn.ReLU(),
                    ConvNorm(n_text_dim*2, n_mel_channels, kernel_size=1,
                             bias=False)
                )

            if self.n_group_size > 1:
                self.unfold_params = {'kernel_size': (n_group_size, 1),
                                      'stride': n_group_size,
                                      'padding': 0, 'dilation': 1}
                self.unfold = nn.Unfold(**self.unfold_params)

            self.exit_steps = []
            self.n_early_size = n_early_size
            n_mel_channels = n_mel_channels*n_group_size
            # form lstm parameter
            self.lstm_applicable = [True if i in self.lstm_applicable_steps
                                    else False for i in range(self.n_flows)]

            for i in range(self.n_flows):
                if i > 0 and i % n_early_every == 0:  # early exitting
                    n_mel_channels -= self.n_early_size
                    self.exit_steps.append(i)

                self.flows.append(FlowStep(
                    n_mel_channels, n_flowstep_cond_dims,
                    n_conv_layers_per_step, affine_model, use_feature_gating,
                    scaling_fn, matrix_decomposition,
                    affine_activation=affine_activation,
                    enable_lstm=self.lstm_applicable[i], p_dropout=p_dropout))

    def encode_speaker(self, spk_ids):
        spk_ids = spk_ids * 0 if self.dummy_speaker_embedding else spk_ids
        spk_vecs = self.speaker_embedding(spk_ids)
        return spk_vecs

    def encode_text(self, text, in_lens):
        # text_embeddings: b x len_text x n_text_dim
        text_embeddings = self.embedding(text).transpose(1, 2)
        # text_enc: b x n_text_dim x encoder_dim (512)
        if in_lens is None:
            text_enc = self.encoder.infer(text_embeddings).transpose(1, 2)
        else:
            text_enc = self.encoder(text_embeddings, in_lens).transpose(1, 2)
        return text_enc, text_embeddings

    def preprocess_context(self, context, speaker_vecs, out_lens=None, f0=None,
                           energy_avg=None):

        if self.n_group_size > 1:
            context = self.unfold(context.unsqueeze(-1))
            # todo(rvalle): fix unfolding zero-padded values
            if f0 is not None:
                f0 = self.unfold(f0[:, None, :, None])
            if energy_avg is not None:
                energy_avg = self.unfold(energy_avg[:, None, :, None])
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, context.shape[2])
        context_w_spkvec = torch.cat((context, speaker_vecs), 1)

        if self.use_context_lstm:
            if self.context_lstm_w_f0_and_energy:
                if f0 is not None:
                    context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

                if energy_avg is not None:
                    context_w_spkvec = torch.cat(
                        (context_w_spkvec, energy_avg), 1)
            unfolded_out_lens = (out_lens // self.n_group_size).long().cpu()
            
            unfolded_out_lens_packed = nn.utils.rnn.pack_padded_sequence(
                context_w_spkvec.transpose(1, 2), unfolded_out_lens,
                batch_first=True, enforce_sorted=False)
            self.context_lstm.flatten_parameters()
            context_lstm_packed_output, _ = self.context_lstm(
                unfolded_out_lens_packed)
            context_lstm_padded_output, dc = nn.utils.rnn.pad_packed_sequence(
                context_lstm_packed_output, batch_first=True)
            context_w_spkvec = context_lstm_padded_output.transpose(1, 2)

        if not self.context_lstm_w_f0_and_energy:
            if f0 is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

            if energy_avg is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, energy_avg), 1)

        return context_w_spkvec

    def fold(self, mel):
        """Inverse of the self.unfold(mel.unsqueeze(-1)) operation used for the
        grouping or "squeeze" operation on input

        Args:
            mel: B x C x T tensor of temporal data
        """
        mel = nn.functional.fold(
            mel, output_size=(mel.shape[2]*self.n_group_size, 1),
            **self.unfold_params).squeeze(-1)
        return mel

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer receive a gradient
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas(attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]], width=1)
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out

    def set_speaker_vec_aug_index(self, speaker_vecs, aug_idxs):
        aug_flags = torch.zeros_like(speaker_vecs)[:, -self.n_aug_dims:]
        if aug_idxs is None:
            aug_flags[:, 0] = 1.0
        else:
            for ind in range(len(speaker_vecs)):
                aug_flags[ind, aug_idxs[ind]] = 1
        speaker_vecs = torch.cat(
            (speaker_vecs[:, :-self.n_aug_dims], aug_flags), dim=1)
        return speaker_vecs

    def forward(self, mel, speaker_ids, text, in_lens, out_lens,
                binarize_attention=False, attn_prior=None, aug_idxs=None,
                f0=None, energy_avg=None, voiced_mask=None, p_voiced=None):
        speaker_vecs = self.encode_speaker(speaker_ids)
        # set the indicator flags to indicate augmentation
        speaker_vecs = self.set_speaker_vec_aug_index(speaker_vecs, aug_idxs)
        text_enc, text_embeddings = self.encode_text(text, in_lens)
        if self.use_text_conditional_priors:
            text_conditional_priors = self.prior_predictor(text_embeddings)

        log_s_list, log_det_W_list, z_mel = [], [], []
        attn = None
        attn_soft = None
        attn_hard = None
        text_conditional_priors_reshaped = None
        if 'atn' in self.what_to_train or 'ftp' in self.what_to_train:
            # make sure to do the alignments before folding
            attn_mask = get_mask_from_lengths(in_lens)[..., None] == 0
            # attn_mask shld be 1 for unsd t-steps in text_enc_w_spkvec tensor
            if self.use_text_conditional_priors:
                attn_soft, attn_logprob = self.attention(
                    mel, text_conditional_priors, out_lens, attn_mask,
                    key_lens=in_lens, keys_encoded=text_enc,
                    attn_prior=attn_prior)
            else:
                attn_soft, attn_logprob = self.attention(
                    mel, text_embeddings, out_lens, attn_mask,
                    key_lens=in_lens, keys_encoded=text_enc,
                    attn_prior=attn_prior)

            if binarize_attention:
                attn = self.binarize_attention(attn_soft, in_lens, out_lens)
                attn_hard = attn
            else:
                attn = attn_soft
            context = torch.bmm(text_enc, attn.squeeze(1).transpose(1, 2))
            if self.use_text_conditional_priors:
                text_conditional_priors_reshaped = self.unfold(
                    torch.bmm(text_conditional_priors,
                              attn.squeeze(1).transpose(1, 2)).unsqueeze(-1))

            if self.zero_out_context:
                context *= 0

        # Forward pass for mel decoder
        if 'ftp' in self.what_to_train:
            if self.n_group_size > 1:
                # might truncate some frames at the end, but that's ok
                # sometimes referred to as the "squeeeze" operation
                # invert this by calling self.fold(mel_or_z)
                mel = self.unfold(mel.unsqueeze(-1))
            z_out = []
            # where context is folded
            # mask f0 in case values are interpolated
            context_w_spkvec = self.preprocess_context(
               context, speaker_vecs, out_lens, f0*voiced_mask, energy_avg)
            log_s_list, log_det_W_list, z_out = [], [], []
            for i, flow_step in enumerate(self.flows):
                if i in self.exit_steps:
                    z = mel[:, :self.n_early_size]
                    z_out.append(z)
                    mel = mel[:, self.n_early_size:]
                mel, log_det_W, log_s = flow_step(mel, context_w_spkvec)
                log_s_list.append(log_s)
                log_det_W_list.append(log_det_W)

            z_out.append(mel)
            z_mel = torch.cat(z_out, 1)

        # duration predictor forward pass
        duration_model_outputs = None
        if 'dpf' in self.what_to_train:
            if attn_hard is None:
                attn_hard = self.binarize_attention(
                    attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            attn_hard_reduced = attn_hard.sum(2)[:, 0, :]
            duration_model_outputs = self.dur_pred_layer(
                torch.detach(text_enc),
                torch.detach(speaker_vecs),
                torch.detach(attn_hard_reduced.float()), in_lens)

        # f0, energy average, p_voiced predictor forward pass
        feature_model_outputs = None
        if 'fev-pf' in self.what_to_train:
            if attn_hard is None:
                attn_hard = self.binarize_attention(
                    attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            attn_hard_reduced = attn_hard.sum(2)[:, 0, :]
            text_enc_time_expanded = self.length_regulator(
                torch.detach(text_enc).transpose(1, 2),
                torch.detach(attn_hard_reduced.float()))
            text_enc_time_expanded = text_enc_time_expanded.transpose(1, 2)

            if self.noise_to_unvoiced_in_f0:
                noise = f0.clone().normal_() * self.noise_to_unvoiced_in_f0
                noise = (-1*voiced_mask + 1) * noise
                f0 += noise

            if self.noise_to_pvoiced:
                noise = p_voiced.clone().normal_() * self.noise_to_pvoiced
                p_voiced += noise

            f0_energy_avg_voiced = torch.cat(
                (f0[:, None], energy_avg[:, None], p_voiced[:, None]), dim=1)
            feature_model_outputs = self.feature_pred_layer(
                text_enc_time_expanded,
                torch.detach(speaker_vecs),
                f0_energy_avg_voiced, out_lens)

        outputs = {'z_mel': z_mel,
                   'log_det_W_list': log_det_W_list,
                   'log_s_list': log_s_list,
                   'duration_model_outputs': duration_model_outputs,
                   'feature_model_outputs': feature_model_outputs,
                   'attn_soft': attn_soft,  # computed soft attention
                   'attn': attn,  # attention that was used (hard or soft)
                   'text_embeddings': text_embeddings,
                   'mean': None,
                   'log_var': None,
                   'prob': None,
                   'attn_logprob': attn_logprob,
                   'text_conditional_priors': text_conditional_priors_reshaped
                   }

        return outputs

    def infer_complete(self, speaker_id, text, sigma, sigma_txt, sigma_feats,
                       token_dur_scaling=1.0, f0_scaling=1.0, f0_mean=0.0,
                       f0_std=0.0, energy_mean=0.0, energy_std=0.0,
                       token_duration_max=100):
        n_tokens = text.shape[1]

        spk_vec = self.encode_speaker(speaker_id)
        spk_vec = self.set_speaker_vec_aug_index(spk_vec, None)

        txt_enc, txt_emb = self.encode_text(text, None)

        # get token duration
        res_dur = torch.cuda.FloatTensor(1, 1, n_tokens)
        res_dur = res_dur.normal_() * sigma_txt

        dur = self.dur_pred_layer.infer(res_dur, txt_enc, spk_vec)[:, 0]
        dur = dur.clamp(0, token_duration_max)
        dur = dur * token_dur_scaling if token_dur_scaling > 0 else dur
        dur = (dur + 0.5).floor().int()
        n_frames = dur.sum().item()
        out_lens = torch.LongTensor([n_frames])

        # get f0 and energy and p voiced
        txt_enc_time_expanded = self.length_regulator(
            txt_enc.transpose(1, 2), dur)
        txt_enc_time_expanded = txt_enc_time_expanded.transpose(1, 2)
        res_feats = torch.cuda.FloatTensor(1, 3, n_frames)
        res_feats = res_feats.normal_() * sigma_feats
        features = self.feature_pred_layer.infer(
            res_feats, txt_enc_time_expanded, spk_vec)
        f0, energy_avg, p_voiced = features[0]
        f0 = f0[None]
        energy_avg = energy_avg[None]
        p_voiced = p_voiced[None]
        if self.feature_pred_layer._get_name() == 'FP':
            p_voiced = torch.sigmoid(p_voiced)
        p_voiced = (p_voiced > 0.5).to(p_voiced.dtype)

        # mean and standard deviation adjustment
        if f0_mean != 0.0 or f0_std != 0.0:
            mask = p_voiced.bool()
            # denorm, shift, scale, norm
            f0[mask] = torch.log(
                shift_and_scale_data(torch.exp(f0[mask]), f0_mean, f0_std)).to(
                    dtype=f0.dtype)

        # f0 scaling, including over time
        if isinstance(f0_scaling, list):
            mask = p_voiced.bool()
            f0_scaling_log = torch.log(
                torch.linspace(f0_scaling[0], f0_scaling[1], mask.sum()))
            f0[mask] = f0[mask] + f0_scaling_log.to(f0.device, dtype=f0.dtype)
        elif f0_scaling != 1.0:
            mask = p_voiced.bool()
            f0_scaling_log = torch.log(torch.tensor([f0_scaling]))
            f0[mask] = f0[mask] + f0_scaling_log.to(f0.device, dtype=f0.dtype)

        f0_masked = f0 * p_voiced
        context = self.length_regulator(
            txt_enc.transpose(1, 2), dur).transpose(1, 2)

        context_w_spkvec = self.preprocess_context(
            context, spk_vec, out_lens, f0_masked, energy_avg)
        out_lens = torch.tensor([context.shape[-1]])

        residual = torch.cuda.FloatTensor(
            1, 80 * self.n_group_size, n_frames // self.n_group_size)
        residual = residual.normal_() * sigma

        exit_steps_stack = self.exit_steps.copy()
        mel = residual[:, len(exit_steps_stack) * self.n_early_size:]
        remaining_residual = residual[:, :len(exit_steps_stack)*self.n_early_size]
        for i, flow_step in enumerate(reversed(self.flows)):
            curr_step = len(self.flows) - i - 1
            mel = flow_step(mel, context_w_spkvec, reverse=True)
            if len(exit_steps_stack) > 0 and curr_step == exit_steps_stack[-1]:
                # concatenate the next chunk of z
                exit_steps_stack.pop()
                residual_to_add = remaining_residual[
                    :, len(exit_steps_stack)*self.n_early_size:]
                remaining_residual = remaining_residual[
                    :, :len(exit_steps_stack)*self.n_early_size]
                mel = torch.cat((residual_to_add, mel), 1)

        if self.n_group_size > 1:
            mel = self.fold(mel)

        return {'mel': mel,
                'dur': dur,
                'f0': f0,
                'f0_masked': f0_masked,
                'energy_avg': energy_avg,
                'p_voiced': p_voiced
                }

    def infer(self, residual, speaker_ids, text, token_durations, f0,
              energy_avg):
        """Inference function. Inverse of the forward pass

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            residual: input residual after flow transformation.
                Technically the mel spectrogram values
        """

        spk_vec = self.encode_speaker(speaker_ids)
        spk_vec = self.set_speaker_vec_aug_index(spk_vec, None)
        text_enc, text_emb = self.encode_text(text, None)
        context = self.length_regulator(
            text_enc.transpose(1, 2), token_durations).transpose(1, 2)
        out_lens = torch.tensor([context.shape[-1]])

        if self.n_group_size > 1:
            # might truncate some frames at the end, but that's ok
            residual = self.unfold(residual[:, :, :out_lens[0]].unsqueeze(-1))

        context_w_spkvec = self.preprocess_context(
                context, spk_vec, out_lens, f0, energy_avg)
        exit_steps_stack = self.exit_steps.copy()
        mel = residual[:, len(exit_steps_stack)*self.n_early_size:]
        remaining_residual = residual[
            :, :len(exit_steps_stack)*self.n_early_size]

        for i, flow_step in enumerate(reversed(self.flows)):
            curr_step = len(self.flows) - i - 1
            mel = flow_step(mel, context_w_spkvec, reverse=True)
            if len(exit_steps_stack) > 0 and curr_step == exit_steps_stack[-1]:
                # concatenate the next chunk of z
                exit_steps_stack.pop()
                residual_to_add = remaining_residual[
                    :, len(exit_steps_stack)*self.n_early_size:]
                remaining_residual = remaining_residual[
                    :, :len(exit_steps_stack)*self.n_early_size]
                mel = torch.cat((residual_to_add, mel), 1)

        if self.n_group_size > 1:
            mel = self.fold(mel)

        return {'mel': mel}

    def infer_token_durations(self, residual, speaker_ids, text, aug_ids=None):
        spk_vec = self.encode_speaker(speaker_ids)
        spk_vec = self.set_speaker_vec_aug_index(spk_vec, aug_ids)
        txt_enc, txt_emb = self.encode_text(text, None)
        dur = self.dur_pred_layer.infer(residual, txt_enc, spk_vec)[:, 0]
        return {'dur': dur}

    def infer_features(self, residual, speaker_ids, text, dur, aug_ids=None):
        spk_vec = self.encode_speaker(speaker_ids)
        spk_vec = self.set_speaker_vec_aug_index(spk_vec, aug_ids)
        txt_enc, txt_emb = self.encode_text(text, None)

        # get f0 and energy and p voiced
        txt_enc_time_expanded = self.length_regulator(
            txt_enc.transpose(1, 2), dur)
        txt_enc_time_expanded = txt_enc_time_expanded.transpose(1, 2)
        features = self.feature_pred_layer.infer(
            residual, txt_enc_time_expanded, spk_vec)

        f0, energy_avg, p_voiced = features[0]
        return {'f0': f0,
                'energy_avg': energy_avg,
                'p_voiced': p_voiced
                }

    def test_invertibility(self, residual, speaker_ids, text, token_durations):
        """Model invertibility check. Call this like you would self.infer()

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of int speaker ids (single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            error: should be in the order of 1e-5 or less, or there may be an
            invertibility bug
        """
        mel = self.infer(residual, speaker_ids, text, token_durations)
        in_lens = torch.LongTensor([text.shape[1]]).cuda()
        residual_recon, log_s_list, _, _, _, _, _ = self.forward(
            mel, speaker_ids, text, token_durations, in_lens, None)
        if self.n_group_size > 1:
            residual_recon = self.fold(residual_recon)
        error = (residual_recon - residual[:, :, :residual_recon.shape[2]]).abs().mean()
        return error

    def test_invertibility_text(self, residual, speaker_ids, text, in_lens):
        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        # text_embeddings: b x len_text x n_text_dim
        text_embeddings = self.embedding(text).transpose(1, 2)
        # text_enc: b x n_text_dim x encoder_dim (512)
        text_enc = self.encoder(text_embeddings, in_lens).transpose(1, 2)

        if self.dummy_speaker_embedding:
            durfn_inputs = torch.detach(text_enc)
        else:
            speaker_vec_dur = speaker_vecs[:, :, None].expand(-1, -1, text_enc.shape[2])
            durfn_inputs = torch.cat((torch.detach(text_enc), speaker_vec_dur), 1)
        token_durations = self.dur_pred_layer.infer(residual, durfn_inputs)
        residual_recon, log_det_W_list, log_s_list = self.dur_pred_layer(
           text_enc, token_durations)
        error = (residual_recon[:, :residual.shape[1]] - residual).abs().mean()
        return error

    def get_context(self, text, token_durations=None):
        text_enc = self.encode_text(text, None)

        if token_durations is None:
            token_durations = self.dur_pred_layer(text_enc)
        context = self.length_regulator(
            text_enc.transpose(1, 2), token_durations).transpose(1, 2)
        return context


    @staticmethod
    def remove_weightnorm(model):
        def remove(conv_list):
            new_conv_list = torch.nn.ModuleList()
            for old_conv in conv_list:
                old_conv = torch.nn.utils.remove_weight_norm(old_conv)
                new_conv_list.append(old_conv)
            return new_conv_list
        ftp = model
        for flow in ftp.flows:
            afp = flow.affine_tfn.affine_param_predictor
            afp.start = torch.nn.utils.remove_weight_norm(afp.start)
            afp.in_layers = remove(afp.in_layers)
            if hasattr(afp, 'cond_layer'):
                afp.cond_layer = torch.nn.utils.remove_weight_norm(afp.cond_layer)
            afp.res_skip_layers = remove(afp.res_skip_layers)
        return ftp

    def remove_norms(self):
        """Removes spectral and weightnorms from model. Call before inference
        """
        for name, module in self.named_modules():
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0')
                logging.info("Removed spectral norm from %s" % {name})
            except:
                pass
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0_reverse')
                logging.info("Removed spectral norm from %s" % {name})
            except:
                pass
            try:
                nn.utils.remove_weight_norm(module)
                logging.info("Removed wnorm from %s" % {name})
            except:
                pass
