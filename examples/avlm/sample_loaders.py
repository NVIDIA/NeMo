import os
import re
import json
import torch

from nemo.collections.multimodal.data.energon.config import (
    ImageToken,
    AudioToken,
    VideoToken,
)
from nemo.collections.avlm.data.energon.avlm_sample_config import MediaDict


def get_media(raw, media_type, value, offset=None, duration=None):
    """
    Return:
        if media_type == 'text', return the text string
        if media_type == 'image', return as PIL Image
        if media_type == 'audio' or 'video', return as MediaDict
    """
    assert media_type in ["text", "audio", "video", "image"]

    if media_type == "text":
        return value
    elif media_type == "audio" or media_type == "video":
        media_dict = {media_type: raw[value]}
        if offset is not None:
            media_dict["offset"] = offset
        if duration is not None:
            media_dict["duration"] = duration
        return MediaDict(**media_dict)
    else:
        return raw[value]


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
        sequence.append(get_media(raw,t,v,offset,duration) for t, v in media if v is not None)

    return dict(__key__=raw["__key__"], 
        sequence=sequence,
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

```json structure 1
{
  "audios": "sample_000001.2345ew.flac,sample_000001.gd1dtg.wav",
  "audio_durations": [5.3058125, 3.06238],
  "videos": "sample_000001.35tags.mp4",
  "video_durations": [5.607625],
  "images": "sample_000001.as23ds.jpg,sample_000001.gds233.jpg",
  "conversations": [
    {
      "from": "User",
      "value": "<audio>"
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
    },
    {
      "from": "User",
      "value": "Describe what is NeMo based on the tutorial video: <video> and the information in the two images: <image> <image>. Combine that information with sound <audio>. Answer: "
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
    }
  ]
}
```

```json structure 2
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
      "duration": [null, 5.607625, null, null, null, null, null, 3.06238, null ],
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


QAMediaTokenTypeMapping = {
    "audio": AudioToken().token_str,
    "video": VideoToken().token_str,
    "image": ImageToken().token_str,
}

def sample_loader_QA(raw: dict) -> dict:
    # Note that only the images are decoded, all other files are read as raw bytes
    jsn = json.loads(raw["json"])
    output_dict = {
        "contexts": [],
        "answers": [],
        "audios": [],
        "videos": [],
        "images": [],
    }


    # process structure 1
    if "audio" in jsn or "audios" in jsn:
        audio_files = jsn["audio"].split(",") if "audio" in jsn else jsn["audios"].split(",")
        offsets = jsn.get("audio_offsets") or [None] * len(audio_files)
        durations = jsn.get("audio_durations") or [None] * len(audio_files)
        if not isinstance(offsets, list):
            offsets = [offsets]
        if not isinstance(durations):
            durations = [durations]
        output_dict["audios"] = [get_media(raw, "audio", f.split(',',1)[1], offsets[i], durations[i]) for i,f in enumerate(audio_files)]

    if "video" in jsn or "videos" in jsn:
        video_files = jsn["video"].split(",") if "video" in jsn else jsn["videos"].split(",")
        offsets = jsn.get("video_offsets") or [None] * len(video_files)
        durations = jsn.get("video_durations") or [None] * len(video_files)
        if not isinstance(offsets, list):
            offsets = [offsets]
        if not isinstance(durations):
            durations = [durations]
        output_dict["videos"] = [get_media(raw, "video", f.split(',',1)[1], offsets[i], durations[i]) for i,f in enumerate(video_files)]

    if "image" in jsn or "images" in jsn:
        image_files = jsn["image"].split(",") if "image" in jsn else jsn["images"].split(",")
        offsets = jsn.get("image_offsets") or [None] * len(image_files)
        durations = jsn.get("image_durations") or [None] * len(image_files)
        if not isinstance(offsets, list):
            offsets = [offsets]
        if not isinstance(durations):
            durations = [durations]
        output_dict["images"] = [get_media(raw, "image", f.split(',',1)[1], offsets[i], durations[i]) for i,f in enumerate(image_files)]

    for turn in jsn["conversations"]:
        if "type" not in turn:
            # process structure 1
            string = turn["value"]
        else:
            # process structure 2
            string = ""
            types = [t.lower() for t in turn["type"].split(",")]
            values = turn["value"]
            if not isinstance(values, list):
                values = [values]

            offsets = turn.get("offset") or [None]*len(values)
            durations = turn.get("duration") or [None]*len(values)
            if not isinstance(offsets, list):
                offsets = [offsets]
            if not isinstance(durations, list):
                durations = [durations]

            for t, v, offset, duration in zip(types, values, offsets, durations):
                raw_media = get_media(t, v, offset, duration)
                if t == "text":
                    string += raw_media
                else:
                    string += QAMediaTokenTypeMapping[t]
                    output_dict[t].append(raw_media)

        if turn["from"].lower() == "assistant" or turn["from"].lower() == "gpt":
            output_dict["answers"].append(string)
        elif turn["from"].lower() == "user" or turn["from"].lower() == "human":
            output_dict["contexts"].append(string)

    return dict(
        contexts=output_dict["contexts"],
        answers=answers if output_dict["answers"] else None,
        audios=audios if output_dict["audios"] else None,
        videos=videos if output_dict["videos"] else None,
        images=images if output_dict["images"] else None,
    )


def part_filter_QA(part: str) -> bool:
    # Need to load all parts
    return True
