from enum import Enum


class PrettyStrEnum(Enum):
    """
    Pretty enum with string values for config options with choices
    """

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value: object):
        choices = ', '.join(map(str, cls))
        raise ValueError(f"{value} is not a valid {cls.__name__}. Possible choices: {choices}")
