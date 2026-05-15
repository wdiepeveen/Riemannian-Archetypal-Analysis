from src.radials.unimodal.centered_ellipsoid.data_enclosing import DataEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.offcentered_ellipsoid.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid import TrimmedEllipsoidRadial

class DataEnclosingTrimmedEllipsoidRadial(TrimmedEllipsoidRadial):
    def __init__(self, data, center, alpha=1.1, beta=0.1, aggregation='softmin'):
        d = data.shape[1]
        ellipsoid_radial = DataEnclosingOffCenteredEllipsoidRadial(data, center, alpha=alpha, beta=beta)
        trim_radial = DataEnclosingCenteredEllipsoidRadial(data, center, alpha=alpha, beta=beta)

        super().__init__(d, ellipsoid_radial, trim_radial, aggregation=aggregation) 