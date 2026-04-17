from src.radials.multimodal import MultiModalRadial
from radials.unimodal.offcentered_ellipsoid.ellipsoid_enclosing import EllipsoidEnclosingOffCenteredEllipsoidRadial
from radials.unimodal.offcentered_ellipsoid.gaussian_enclosing import GaussianEnclosingOffCenteredEllipsoidRadial

class MultiOffCenteredEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, mus, c=4/3, p=None, aggregation='max'):
        if p is None:
            radials = [EllipsoidEnclosingOffCenteredEllipsoidRadial(cov, mu, c=c) for cov, mu in zip(covs, mus)]
        else:
            radials = [GaussianEnclosingOffCenteredEllipsoidRadial(cov, mu, c=c, p=p) for cov, mu in zip(covs, mus)]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)