from src.distributions.stars import StarDistribution
from src.radials.unimodal.elliposoid.centered.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.centered.gaussian_enclosing import GaussianEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.ellipsoid_enclosing import EllipsoidEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.gaussian_enclosing import GaussianEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.ellipsoid_enclosing_offcentered import EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.gaussian_enclosing_offcentered import GaussianEnclosingOffCenteredTrimmedEllipsoidRadial

class UnimodalEllipsoidStarDistribution(StarDistribution):
    def __init__(self, cov, mu=None, p=None, trimmed=False):
        if mu is None:
            if p is None:
                radial = EllipsoidEnclosingCenteredEllipsoidRadial(cov)
            else:
                radial = GaussianEnclosingCenteredEllipsoidRadial(cov, p=p)
        else:
            if p is None:
                if not trimmed:
                    radial = EllipsoidEnclosingOffCenteredEllipsoidRadial(cov, mu)
                else:
                    radial = EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu)
            else:
                if not trimmed:
                    radial = GaussianEnclosingOffCenteredEllipsoidRadial(cov, mu, p=p)
                else:
                    radial = GaussianEnclosingOffCenteredTrimmedEllipsoidRadial(cov, mu, p=p)
        d = cov.shape[0]
        super().__init__(d, radial)