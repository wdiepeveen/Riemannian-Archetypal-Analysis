from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.elliposoid.centered.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.centered.gaussian_enclosing import GaussianEnclosingCenteredEllipsoidRadial

class MultiCenteredEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, p=None, aggregation='sum'):
        if p is None:
            radials = [EllipsoidEnclosingCenteredEllipsoidRadial(cov) for cov in covs]
        else:
            radials = [GaussianEnclosingCenteredEllipsoidRadial(cov, p=p) for cov in covs]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)