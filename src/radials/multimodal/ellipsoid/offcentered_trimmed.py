from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.trimmed_ellipsoid.ellipsoid_enclosing_offcentered import EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.gaussian_enclosing_offcentered import GaussianEnclosingOffCenteredTrimmedEllipsoidRadial

class MultiOffCenteredTrimmedEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, mus, p=None, aggregation='max'):
        if p is None:
            radials = [EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu, c=4/3) for cov, mu in zip(covs, mus)]
        else:
            radials = [GaussianEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu, c=4/3, p=p) for cov, mu in zip(covs, mus)]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)