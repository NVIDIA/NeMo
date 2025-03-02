import json
import torch

def get_media(t, v, offset=None, duration=None):
    """
    Return:
        if t == 'text', return the text string
        if t == 'image', return as PIL Image
        if t == 'audio' or 'video', return as dict with keys: type, value in bytes, offset: optional, duration: optional
    """
    assert t in ["text", "audio", "video", "image"]

    if t == "text":
        return v
    elif t == "audio" or t == "video":
        media_dict = {t: json[v]}
        if offset is not None:
            media_dict["offset"] = offset
        if duration is not None:
            media_dict["duration"] = duration
        return media_dict
    else:
        return json[v]

"""
(interleaved-sample-loader)=
## Example: Interleaved Data and Arbitrary Media Count

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
    "duration": [null, null, null],
    "offset": [null, null, null],
    "images": [null, "2345ew.jpg", null],
    "texts": ["This is some text, an image is following.", null, "More text after the image."],
}
```
Note that the image path corresponds to the filename of the image after the first "." in the sample. This is all part of the extension as defined by webdataset. Everything before the first "." is part of the sample key and must be equal to match into the same group.
"""

def sample_loader_interleaved(raw: dict) -> dict:
    # Note that only the images are decoded, all other files are read as raw bytes.
    jsn = json.loads(raw["json"])
    sequence = []
    for text, audio, video, image, offset, duration in zip(
        jsn["texts"],
        jsn.get("audios") or [None]*len(jsn["texts"]), 
        jsn.get("videos") or [None]*len(jsn["texts"]), 
        jsn["images"],        
        jsn.get("offset") or [None]*len(jsn["texts"]), 
        jsn.get("duration") or [None]*len(jsn["texts"]), 
    ):
        media = [("text", text), ("audio", audio), ("video", video), ("image", image)]
        sequence.append(get_media(t,v,offset,duration) for t, v in media if v is not None)

    return dict(__key__=raw["__key__"], 
        contexts=[sequence],
    )


def part_filter_interleaved(part: str) -> bool:
    # Need to load all parts
    return True


"""
(multi-turn-sample-loader)=
## Example: Interleaved Data and Arbitrary Media Count

### The Webdataset Structure

The structure of the shard files could be like this:

`tar -tvf shard_0.tar`:
```python
sample_000001.2345ew.flac
sample_000001.35tags.mp4
sample_000001.as23ds.jpg
sample_000001.gd1dtg.wav
sample_000001.gds233.jpg
sample_000001.json
sample_000002.asf234.wav
sample_000002.json
sample_000003.json
```

```json
{
  "conversations": [
    {
      "type": "audio",
      "from": "User",
      "duration": 5.3058125,
      "value": "2345ew.flac"
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
    },
    {
      "type": "text, video, text, image, text, image, text, audio, text",
      "from": "User",
      "duration": [null, 5.607625, null, null, null, null, null, , null ],
      "value": ["Describe what is NeMo based on the tutorial video: ", 
        "35tags.mp4", 
        " and the information in the two images: ", 
        "as23ds.jpg", 
        " ",
        "gds233.jpg", 
        ". Combine that information with sound ",
        "gd1dtg.wav",
        ". Answer: ",
      ]
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
    }
  ]
}
```
"""

def sample_loader_multiturn(raw: dict) -> dict:
    # Note that only the images are decoded, all other files are read as raw bytes
    jsn = json.loads(raw["json"])
    contexts = []
    answers = []

    for turn in jsn["conversations"]:
        sequence = []
        
        if isinstance(turn["value"], str):
            sequence.append(get_media(turn["type"], turn["value"], turn.get("offset"), turn.get("duration")))
        else:
            for t, v, offset, duration in zip(
                turn["type"].split(","), 
                turn["value"],
                turn.get("offset") or [None]*len(turn["value"]),
                turn.get("duration") or [None]*len(turn["value"])
            ):
                sequence.append(get_media(t,v, offset, duration))

        if turn["from"] == "Assistant":
            answers.append(sequence)
        elif turn["from"] == "User":
            contexts.append(sequence)

    return dict(__key__=raw["__key__"], 
        contexts=sequence,
        answers=answers,
    )


def part_filter_multiturn(part: str) -> bool:
    # Need to load all parts
    return True
