"""
(interleaved-sample-loader)=
## Example: Interleaved Data and Arbitrary Image Count

### The Webdataset Structure

If you need multiple files with an arbitrary number of data per sample, e.g. multiple image / video / audio files, this shows a blueprint for how to setup your webdataset tar files and how to load that webdataset with Energon.

The structure of the shard files could be like this:

`tar -tvf shard_0.tar`:
```python
sample_000001.2345ew.jpg
sample_000001.json
sample_000002.35tags.mp4
sample_000002.as23ds.jpg
sample_000002.gd1dtg.wav
sample_000002.gds233.jpg
sample_000002.json
sample_000002.sdag42.jpg
sample_000003.json
sample_000004.asf234.wav
sample_000004.json
```

where the structure of a json file is:

`tar -xf shard_0.tar sample_000001.json -O`:
```json
{
    "audios": [null, null, null],
    "videos": [null, null, null],
    "images": [null, "2345ew.jpg", null],
    "texts": ["This is some text, an image is following.", null, "More text after the image."],
}
```
Note that the image path corresponds to the filename of the image after the first "." in the sample. This is all part of the extension as defined by webdataset. Everything before the first "." is part of the sample key and must be equal to match into the same group.
"""

import torch


def sample_loader_interleaved(raw: dict) -> dict:
    # Note that the images are already decoded, as well as the json part.
    sequence = []
    for media in zip(
        raw["json"]["audios"], 
        raw["json"]["videos"], 
        raw["json"]["images"], 
        raw["json"]["texts"]
    ):
        sequence.extend([m for m in media if m is not None])

    return dict(__key__=raw["__key__"], sequence=sequence,)


def part_filter_interleaved(part: str) -> bool:
    # Need to load all parts
    return True
