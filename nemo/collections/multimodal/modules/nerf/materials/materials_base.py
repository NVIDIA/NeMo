from enum import Enum
from typing import Literal, Optional

from torch import nn


class ShadingEnum(str, Enum):
    TEXTURELESS = "textureless"
    NORMAL = "normal"
    LAMBERTIAN = "lambertian"
    PHONG = "phong"

    # TODO(ahmadki):
    # Oren–Nayar
    # Minnaert
    # Cook–Torrance
    # Ward anisotropic
    # Hanrahan–Krueger
    # Cel shading
    # Gooch shading


class MaterialsBase(nn.Module):
    """
    Base class for materials.
    """

    def __init__(self):
        super(MaterialsBase, self).__init__()
