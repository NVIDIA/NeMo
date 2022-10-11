# let's import all supported processors here to simplify target specification
from .asr_inference import ASRInference
from .create_initial_manifest.create_initial_manifest_mls import CreateInitialManifestMLS
from .modify_manifest.data_to_data import (
    InsIfASRInsertion,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
    SubSubstringToSpace,
    SubSubstringToSubstring,
)
from .modify_manifest.data_to_dropbool import (
    DropASRErrorBeginningEnd,
    DropHighCER,
    DropHighLowCharrate,
    DropHighLowDuration,
    DropHighLowWordrate,
    DropHighWER,
    DropIfRegexInAttribute,
    DropIfSubstringInAttribute,
    DropIfSubstringInInsertion,
    DropIfTextIsEmpty,
    DropLowWordMatchRate,
    DropNonAlphabet,
)
from .write_manifest import WriteManifest
