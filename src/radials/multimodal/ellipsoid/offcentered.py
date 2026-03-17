from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.elliposoid.offcentered.ellipsoid_enclosing import EllipsoidEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.gaussian_enclosing import GaussianEnclosingOffCenteredEllipsoidRadial

class MultiOffCenteredEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, mus, p=None, aggregation='max'):
        if p is None:
            radials = [EllipsoidEnclosingOffCenteredEllipsoidRadial(cov, mu) for cov, mu in zip(covs, mus)]
        else:
            radials = [GaussianEnclosingOffCenteredEllipsoidRadial(cov, mu, p=p) for cov, mu in zip(covs, mus)]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)