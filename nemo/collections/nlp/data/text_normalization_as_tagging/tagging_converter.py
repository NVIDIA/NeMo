# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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


from nemo.collections.nlp.data.text_normalization_as_tagging.tagging import Tag, EditingTask
from typing import List


class TaggingConverterTrivial(object):
    """Converter from training target texts into tagging format."""

    def __init__(self):
        pass

    def compute_tags(self, task: EditingTask, target: str) -> List[Tag]:
        """Computes tags needed for converting the source into the target.

        Args:
            task: tagging.EditingTask that specifies the input.
            target: Target text.

        Returns:
            List of tagging.Tag objects.
        """
        target_tokens = target.split(" ")
        assert len(target_tokens) == len(task.source_tokens), "len mismatch: " + str(task.source_tokens) + "\n" + target
        tags = []
        for t in target_tokens:
            if t == "<SELF>":
                tags.append(Tag('KEEP'))
            elif t == "<DELETE>":
                tags.append(Tag('DELETE'))
            else:
                tags.append(Tag('DELETE|' + t))
        return tags
