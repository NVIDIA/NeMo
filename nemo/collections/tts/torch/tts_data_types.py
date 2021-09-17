class TTSDataType:
    name = None


class WithLens(TTSDataType):
    """Represent that this TTSDataType returns lengths for data"""


class Audio(WithLens):
    name = "audio"


class Text(WithLens):
    name = "text"


class LogMel(WithLens):
    name = "log_mel"


class Durations(TTSDataType):
    name = "durations"


class DurationPrior(TTSDataType):
    name = "duration_prior"


class Pitch(WithLens):
    name = "pitch"


class Energy(WithLens):
    name = "energy"


MAIN_DATA_TYPES = [Audio, Text]
VALID_SUPPLEMENTARY_DATA_TYPES = [LogMel, Durations, DurationPrior, Pitch, Energy]
DATA_STR2DATA_CLASS = {d.name: d for d in MAIN_DATA_TYPES + VALID_SUPPLEMENTARY_DATA_TYPES}
