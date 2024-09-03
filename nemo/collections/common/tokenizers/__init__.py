# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial
from ..module_import_proxy import ModuleImportProxy
namespace = globals()
lazy_import = partial(ModuleImportProxy, global_namespace=namespace)

lazy_import("nemo.collections.common.tokenizers.aggregate_tokenizer", "AggregateTokenizer")
lazy_import("nemo.collections.common.tokenizers.bytelevel_tokenizers", "ByteLevelTokenizer")
lazy_import("nemo.collections.common.tokenizers.canary_tokenizer", "CanaryTokenizer")
lazy_import("nemo.collections.common.tokenizers.char_tokenizer", "CharTokenizer")
lazy_import("nemo.collections.common.tokenizers.huggingface",  "AutoTokenizer")
lazy_import("nemo.collections.common.tokenizers.regex_tokenizer", "RegExTokenizer")
lazy_import("nemo.collections.common.tokenizers.sentencepiece_tokenizer", "SentencePieceTokenizer")
lazy_import("nemo.collections.common.tokenizers.tiktoken_tokenizer", "TiktokenTokenizer")
lazy_import("nemo.collections.common.tokenizers.tokenizer_spec", "TokenizerSpec")
lazy_import("nemo.collections.common.tokenizers.word_tokenizer", "WordTokenizer")


__all__ = [
    "AggregateTokenizer",
    "ByteLevelTokenizer",
    "CanaryTokenizer",
    "CharTokenizer",
    "AutoTokenizer",
    "RegExTokenizer",
    "SentencePieceTokenizer",
    "TokenizerSpec",
    "WordTokenizer",
]
