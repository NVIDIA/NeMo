# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os

import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling import MegatronGPTPSoftPromptModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="soft_prompt_eval_test")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    model = MegatronGPTPSoftPromptModel.restore_from(cfg.model.restore_path, trainer=trainer)
    model.freeze()

    # has to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # length_params: LengthParam = {
    #     "max_length": cfg.inference.tokens_to_generate,
    #     "min_length": cfg.inference.min_tokens_to_generate,
    # }

    # sampling_params: SamplingParam = {
    #     "use_greedy": cfg.inference.greedy,
    #     "temperature": cfg.inference.temperature,
    #     "top_k": cfg.inference.top_k,
    #     "top_p": cfg.inference.top_p,
    #     "repetition_penalty": cfg.inference.repetition_penalty,
    #     "add_BOS": cfg.inference.add_BOS,
    #     "all_probs": cfg.inference.all_probs,
    #     "compute_logprob": cfg.inference.compute_logprob,
    # }
    prompts = [{"taskname": "squad", "text": "The brief peace in Europe allowed Napoleon to focus on the French colonies abroad. Saint-Domingue had managed to acquire a high level of political autonomy during the Revolutionary Wars, with Toussaint Louverture installing himself as de facto dictator by 1801. Napoleon saw his chance to recuperate the formerly wealthy colony when he signed the Treaty of Amiens. During the Revolution, the National Convention voted to abolish slavery in February 1794. Under the terms of Amiens, however, Napoleon agreed to appease British demands by not abolishing slavery in any colonies where the 1794 decree had never been implemented. The resulting Law of 20 May never applied to colonies like Guadeloupe or Guyane, even though rogue generals and other officials used the pretext of peace as an opportunity to reinstate slavery in some of these places. The Law of 20 May officially restored the slave trade to the Caribbean colonies, not slavery itself. Napoleon sent an expedition under General Leclerc designed to reassert control over Sainte-Domingue. Although the French managed to capture Toussaint Louverture, the expedition failed when high rates of disease crippled the French army. In May 1803, the last 8000 French troops left the island and the slaves proclaimed an independent republic that they called Ha\u00efti in 1804. Seeing the failure of his colonial efforts, Napoleon decided in 1803 to sell the Louisiana Territory to the United States, instantly doubling the size of the U.S. The selling price in the Louisiana Purchase was less than three cents per acre, a total of $15 million.\nWhat was the name of the French general who led the forces that attempted to regain control of Sainte-Domingue?\n", "answer": "Leclerc"},
               {"taskname": "squad", "text": "During the Consulate, Napoleon faced several royalist and Jacobin assassination plots, including the Conspiration des poignards (Dagger plot) in October 1800 and the Plot of the Rue Saint-Nicaise (also known as the Infernal Machine) two months later. In January 1804, his police uncovered an assassination plot against him that involved Moreau and which was ostensibly sponsored by the Bourbon family, the former rulers of France. On the advice of Talleyrand, Napoleon ordered the kidnapping of the Duke of Enghien, violating the sovereignty of Baden. The Duke was quickly executed after a secret military trial, even though he had not been involved in the plot. Enghien's execution infuriated royal courts throughout Europe, become one of the contributing political factors for the outbreak of the Napoleonic Wars.\nWhat was the name of the assassination plot against Napoleon also known as the Infernal Machine?\n", "answer": "the Plot of the Rue Saint-Nicaise"},  
               {"taskname": "squad", "text": "One of the claimants of the English throne opposing William the Conqueror, Edgar Atheling, eventually fled to Scotland. King Malcolm III of Scotland married Edgar's sister Margaret, and came into opposition to William who had already disputed Scotland's southern borders. William invaded Scotland in 1072, riding as far as Abernethy where he met up with his fleet of ships. Malcolm submitted, paid homage to William and surrendered his son Duncan as a hostage, beginning a series of arguments as to whether the Scottish Crown owed allegiance to the King of England.\nWho was the hostage?\n", "answer": "Duncan"}]
    # first method of running text generation, call model.generate method
    response = model.generate(
        inputs=prompts, length_params=None, sampling_params=None
    )

    print("***************************")
    print(response)
    print("***************************")

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
