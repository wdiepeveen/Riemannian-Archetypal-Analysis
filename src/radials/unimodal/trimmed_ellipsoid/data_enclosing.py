from src.radials.unimodal.centered_ellipsoid.data_enclosing import DataEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.offcentered_ellipsoid.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid import TrimmedEllipsoidRadial

class DataEnclosingTrimmedEllipsoidRadial(TrimmedEllipsoidRadial):
    def __init__(self, data, center, c=1.1, reg_param=1e-2, aggregation='softmin'):
        d = data.shape[1]
        ellipsoid_radial = DataEnclosingOffCenteredEllipsoidRadial(data, center, c=c, reg_param=reg_param)
        trim_radial = DataEnclosingCenteredEllipsoidRadial(data, center, c=c, reg_param=reg_param)

        super().__init__(d, ellipsoid_radial, trim_radial, aggregation=aggregation) 