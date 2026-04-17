import torch

from src.radials.unimodal.elliposoid.centered.data_enclosing import DataEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid import TrimmedEllipsoidRadial

class DataEnclosingOffCenteredTrimmedEllipsoidRadial(TrimmedEllipsoidRadial):
    def __init__(self, data, mu, c=1., spherical=True, softmin=False):
        d = data.shape[1]
        ellipsoid_radial = DataEnclosingOffCenteredEllipsoidRadial(data, mu, c=c)
        trim_radial = DataEnclosingCenteredEllipsoidRadial(data)

        super().__init__(d, ellipsoid_radial, trim_radial, softmin=softmin) 