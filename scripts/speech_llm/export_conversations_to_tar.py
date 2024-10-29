from random import Random

import click
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationTarWriter,
)


@click.command()
@click.argument("manifest", type=click.Path())
@click.argument("output_dir", type=click.Path())
@click.option("-n", "--shard_size", type=int, default=100, help="Number of conversations per shard.")
@click.option("--shuffle/--no-shuffle", default=False, help="Shuffle conversations.")
@click.option("-s", "--seed", type=int, default=42, help="Random seed.")
def export(manifest: str, output_dir: str, shard_size: int, shuffle: bool, seed: int):
    with NeMoMultimodalConversationTarWriter(output_dir, shard_size=shard_size) as writer:
        source = NeMoMultimodalConversationJsonlAdapter(manifest, audio_locator_tag="<dummy>")
        if shuffle:
            source = CutSet(source).shuffle(buffer_size=50000, rng=Random(seed))
        for item in source:
            writer.write(item)


if __name__ == '__main__':
    export()
