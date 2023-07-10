# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.multimodal.models.stable_diffusion.samplers import Sampler
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    log_prob_isotropic_gaussian,
    log_prob_gaussian,
)


class AbstractBaseSampler(ABC):
    def __init__(self, model, sampler, schedule="linear", supports_logprobs=False, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        assert isinstance(sampler, Sampler), "Sampler should be of ENUM type Sampler"
        self.sampler = sampler
        self.supports_logprobs = supports_logprobs

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(torch.cuda.current_device())
        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @abstractmethod
    def p_sampling_fn(self):
        pass
    
    def scoring_fn(self):
        pass

    def dpm_sampling_fn(self):
        pass

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        return_logprobs=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for sampling is {size}, eta {eta}")

        if self.sampler is Sampler.DPM:
            return self.dpm_sampling_fn(
                shape=shape,
                steps=S,
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
                x_T=x_T,
            )

        if eta == 0.0 and return_logprobs:
            assert self.sampler is not Sampler.DDIM, 'DDIM eta=0 is deterministic. Logprobs don\'t mean anything'
            # TODO: remove?
            print(f'WARNING. Using eta=0.0 with return_logprobs. Depending on your sampler, logprobs may be meaningless')

        samples, intermediates, logprobs = self.sampling_fn(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            return_logprobs=return_logprobs,
        )
        return samples, intermediates, logprobs

    @torch.no_grad()
    def sampling_fn(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        return_logprobs=False,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, generator=self.model.rng, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        intermediates = {"x_inter": [img], "pred_x0": [img]}
        logprobs = []

        # TODO: Is this needed
        if self.sampler is Sampler.PLMS:
            time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        else:
            time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running {self.sampler.name} Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc=f"{self.sampler.name} Sampler", total=total_steps)
        old_eps = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if self.sampler is Sampler.PLMS:
                ts_next = torch.full(
                    (b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,
                )
            else:
                old_eps = None
                ts_next = None
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img
            if self.supports_logprobs:
                logprob_args = {'return_logprobs': return_logprobs}
            else:
                logprob_args = {}
            outs = self.p_sampling_fn(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
                **logprob_args
            )
            img, pred_x0 = outs[0], outs[1]
            if self.sampler is Sampler.PLMS:
                e_t = outs[2]
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
            if return_logprobs:
                logprobs.append(outs[-1])
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
        return img, intermediates, logprobs

    def _get_model_output(
        self, x, t, unconditional_conditioning, unconditional_guidance_scale, score_corrector, c, corrector_kwargs,
    ):
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
        elif isinstance(c, dict):
            raise NotImplementedError
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        return e_t

    def _get_x_prev_and_pred_x0(
        self,
        use_original_steps,
        b,
        index,
        device,
        x,
        e_t,
        quantize_denoised,
        repeat_noise,
        temperature,
        noise_dropout,
        return_logprobs,
    ):
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
        raw_noise = noise_like(x.shape, device, repeat_noise)
        noise = sigma_t * raw_noise * temperature
        step_std = sigma_t * temperature * torch.ones_like(x)
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev_mean = a_prev.sqrt() * pred_x0 + dir_xt
        x_prev = x_prev_mean + noise
        if return_logprobs:
            return x_prev, pred_x0, log_prob_gaussian(x_prev, mean=x_prev_mean, cov_diag=step_std) #log_prob_isotropic_gaussian(raw_noise)
        return x_prev, pred_x0
    
    def _get_step_logprob(
            self,
            use_original_steps,
            b,
            index,
            device,
            x,
            x_prev, # x at a previous timestep, of which we are trying to find the logprob
            e_t,
            quantize_denoised,
            temperature,
    ):
        """
        Given a noisy image at timestep `index` and a next step less noisy image at timestep
        `index`-1, this function returns the logprob of the model taking that step. 
        Assumes inputs are batched at dim 0.
        """
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        # a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_t = torch.tensor(alphas[index], device=device).reshape((b, 1, 1, 1))
        # a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        a_prev = torch.tensor(alphas_prev[index], device=device).reshape((b, 1, 1, 1))
        # sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sigma_t = torch.tensor(sigmas[index], device=device).reshape((b, 1, 1, 1))
        # sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        sqrt_one_minus_at = torch.tensor(sqrt_one_minus_alphas[index], device=device).reshape((b, 1, 1, 1))
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t

        step_std = sigma_t * temperature * torch.ones_like(x_prev)
        x_prev_mean = a_prev.sqrt() * pred_x0 + dir_xt
        x_prev_standard = (x_prev - x_prev_mean) / step_std
        logprobs = log_prob_gaussian(x_prev, mean=x_prev_mean, cov_diag=step_std)
        # logprobs = log_prob_isotropic_gaussian(x_prev_standard)
        return logprobs