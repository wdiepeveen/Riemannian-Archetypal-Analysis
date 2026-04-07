from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.trimmed_ellipsoid.ellipsoid_enclosing_offcentered import EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.gaussian_enclosing_offcentered import GaussianEnclosingOffCenteredTrimmedEllipsoidRadial

class MultiOffCenteredTrimmedEllipsoidRadial(MultiModalRadial):
    def __init__(self, covs, mus, c=4/3, p=None, aggregation='max', soft_trim_aggregation=False):
        if p is None:
            radials = [EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu, c=c, softmin=soft_trim_aggregation) for cov, mu in zip(covs, mus)]
        else:
            radials = [GaussianEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu, c=c, p=p, softmin=soft_trim_aggregation) for cov, mu in zip(covs, mus)]
        super().__init__(d=covs[0].shape[0], radials=radials, aggregation=aggregation)