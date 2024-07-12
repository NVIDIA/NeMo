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

import argparse
import sys

from nemo.deploy.multimodal import MediaType, NemoQueryMultimodalPyTorch


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Query Triton Multimodal server running in-framework model",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model_name", required=True, type=str, help="Name of the triton model")
    parser.add_argument("-int", "--input_text", required=True, type=str, help="Input text")
    parser.add_argument("-im", "--input_media", required=True, type=str, help="File path of input media")
    parser.add_argument("-mt", "--media_type", required=True, type=str, help="Type the input media (image or video)")
    parser.add_argument("-mol", "--max_output_len", default=128, type=int, help="Max output token length")
    parser.add_argument("-tk", "--top_k", default=1, type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", default=0.0, type=float, help="top_p")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument("-it", "--init_timeout", default=60.0, type=float, help="init timeout for the triton server")

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    nq = NemoQueryMultimodalPyTorch(url=args.url, model_name=args.model_name)
    if args.media_type == "image":
        media_type = MediaType.IMAGE
    elif args.media_type == "video":
        media_type = MediaType.VIDEO
    else:
        raise ValueError("media_type must be either 'image' or 'video'")

    output = nq.query(
        prompts_list=[args.input_text],
        media_filepaths_list=[args.input_media],
        media_type=media_type,
        max_output_len=args.max_output_len,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        init_timeout=args.init_timeout,
    )
    print(output)
