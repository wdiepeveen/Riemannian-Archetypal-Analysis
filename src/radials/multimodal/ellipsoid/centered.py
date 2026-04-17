from src.radials.multimodal import MultiModalRadial
from radials.unimodal.centered_ellipsoid.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial
from radials.unimodal.centered_ellipsoid.gaussian_enclosing import GaussianEnclosingCenteredEllipsoidRadial

class MultiCenteredEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, p=None, aggregation='max'):
        if p is None:
            radials = [EllipsoidEnclosingCenteredEllipsoidRadial(cov) for cov in covs]
        else:
            radials = [GaussianEnclosingCenteredEllipsoidRadial(cov, p=p) for cov in covs]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)