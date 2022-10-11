# let's import all supported processors here to simplify target specification
from .create_initial_manifest.create_initial_manifest_mls import CreateInitialManifestMLS
from .asr_inference import ASRInference
from .write_manifest import WriteManifest
from .modify_manifest.data_to_data import (
    SubSubstringToSpace,
    SubSubstringToSubstring,
    InsIfASRInsertion,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
)

from .modify_manifest.data_to_dropbool import (
    DropHighLowCharrate,
    DropHighLowWordrate,
    DropHighLowDuration,
    DropNonAlphabet,
    DropASRErrorBeginningEnd,
    DropHighCER,
    DropHighWER,
    DropLowWordMatchRate,
    DropIfSubstringInAttribute,
    DropIfRegexInAttribute,
    DropIfSubstringInInsertion,
    DropIfTextIsEmpty,
)

