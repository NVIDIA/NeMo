import torchvision
import numpy as np
import tempfile
import os

class VideoFeaturizer(object):
    def __init__(self):
        pass

    def process(
        self,
        video_file,
        offset,
        duration
    ):

        # Load Video
        video = self.from_file(video_file, offset=offset, duration=duration)

        return video
        
    def from_file(self, video_file, offset, duration):
        
        # Load from filename
        if isinstance(video_file, str):
            video, audio, infos = torchvision.io.read_video(video_file, start_pts=offset, end_pts=offset + duration, pts_unit="sec")

        # Load from bytes
        elif isinstance(video_file, bytes):

            # webdataset.torch_video
            with tempfile.TemporaryDirectory() as dirname:
                fname = os.path.join(dirname, f"file.mp4")
                with open(fname, "wb") as stream:
                    stream.write(video_file)
                    video, audio, infos = torchvision.io.read_video(fname, start_pts=offset, end_pts=offset + duration, pts_unit="sec")
        else:
            raise Exception("Unknown video data format")

        return video
