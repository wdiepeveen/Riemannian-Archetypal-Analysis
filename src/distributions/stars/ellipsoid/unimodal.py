from src.distributions.stars import StarDistribution
from src.radials.unimodal.elliposoid.centered.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.centered.gaussian_enclosing import GaussianEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.ellipsoid_enclosing import EllipsoidEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.gaussian_enclosing import GaussianEnclosingOffCenteredEllipsoidRadial

class UnimodalEllipsoidStarDistribution(StarDistribution):
    def __init__(self, cov, mu=None, p=None):
        if mu is None:
            if p is None:
                radial = EllipsoidEnclosingCenteredEllipsoidRadial(cov)
            else:
                radial = GaussianEnclosingCenteredEllipsoidRadial(cov, p=p)
        else:
            if p is None:
                radial = EllipsoidEnclosingOffCenteredEllipsoidRadial(cov, mu)
            else:
                radial = GaussianEnclosingOffCenteredEllipsoidRadial(cov, mu, p=p)
        d = cov.shape[0]
        super().__init__(d, radial)