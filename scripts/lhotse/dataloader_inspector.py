import click
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from tqdm.auto import tqdm

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import (
    AggregateTokenizer,
    CanaryTokenizer,
    SentencePieceTokenizer,
    TokenizerSpec,
)


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def load_tokenizer(paths: list[str], langs: list[str] = None) -> TokenizerSpec:
    if len(paths) == 1:
        tok = SentencePieceTokenizer(paths[0])
    else:
        assert langs is not None and len(paths) == len(
            langs
        ), f"Cannot create AggregateTokenizer; each tokenizer must have assigned a language via --langs option (we got --tokenizers={paths} and --langs={langs})"
        if any(l == "spl_tokens" for l in langs):
            tokcls = CanaryTokenizer
        else:
            tokcls = AggregateTokenizer
        tok = tokcls({lang: SentencePieceTokenizer(p) for lang, p in zip(langs, paths)})
    return tok


@click.command(context_settings={"show_default": True})
@click.option("-n", "--num-steps", type=int, default=1000, help="Number of steps to iterate dataloader for.")
@click.option(
    "-p",
    "--print-interval",
    type=int,
    default=100,
    help="Number of steps indicating how often the iterated examples should be printed.",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to a training config for a NeMo model.",
)
@click.option(
    "-x",
    "--ds-xpath",
    default="model.train_ds",
    type=str,
    help="Xpath in the config used to find the dataset definition.",
)
@click.option(
    "-t",
    "--tokenizer",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a training config for a NeMo model.",
)
@click.option(
    "-r",
    "--global-rank",
    default=0,
    type=int,
    help="DDP global rank to be set for the dataloader. Useful for investigating dataloading in DDP setups.",
)
@click.option("-w", "--world-size", default=1, type=int, help="DDP world size to be set for the dataloader.")
@click.option(
    "-t",
    "--tokenizer",
    multiple=True,
    help="Path to one or more SPE tokenizers. More than one means we'll use AggregateTokenizer and --langs argument must also be used. When provided, we'll estimate a 2D distribution for input and output sequence lengths.",
)
@click.option(
    "-a",
    "--langs",
    multiple=True,
    help="Language names for each of AggregateTokenizer sub-tokenizers. Only required if tokenizer received two or more paths.",
)
def iterate_dataloader(
    num_steps: int,
    print_interval: int,
    config_path: str,
    ds_xpath: str,
    global_rank: int,
    world_size: int,
    tokenizer: list[str] | None,
    langs: list[str] | None,
):
    cfg = OmegaConf.load(config_path)
    keys = ds_xpath.split(".")
    try:
        for k in keys:
            cfg = cfg[k]
    except ConfigKeyError as e:
        raise KeyError(f"Config key '{k}' from {ds_xpath=} not found in config at {config_path}") from e

    tok = None
    if tokenizer is not None:
        tok = load_tokenizer(tokenizer, langs)

    dloader = get_lhotse_dataloader_from_config(
        cfg, global_rank=global_rank, world_size=world_size, dataset=Identity(), tokenizer=tok
    )

    for idx, item in enumerate(tqdm(dloader)):
        if idx % print_interval == 0:
            print(f"Step {idx}: {item}")
        if idx == num_steps:
            return


if __name__ == '__main__':
    iterate_dataloader()
