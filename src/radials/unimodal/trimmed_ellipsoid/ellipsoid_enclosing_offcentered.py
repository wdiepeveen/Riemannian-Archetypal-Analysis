import torch

from src.radials.unimodal.elliposoid.centered.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.ellipsoid_enclosing import EllipsoidEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.elliposoid.offcentered.gaussian_enclosing import GaussianEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid import TrimmedEllipsoidRadial

class EllipsoidEnclosingOffCenteredTrimmedEllipsoidRadial(TrimmedEllipsoidRadial):
    def __init__(self, cov, mu, c=4/3, spherical=True, softmin=False):
        d = cov.shape[0]
        ellipsoid_radial = EllipsoidEnclosingOffCenteredEllipsoidRadial(cov, mu, c=c)
        Sigma_trim = GaussianEnclosingOffCenteredEllipsoidRadial(cov, mu, c=c).Sigma
        if not spherical:
            trim_radial = EllipsoidEnclosingCenteredEllipsoidRadial(Sigma_trim)
        else:
            lambda_mu = torch.einsum('i,ij,j->', mu, Sigma_trim, mu) / torch.dot(mu, mu)
            trim_radial = EllipsoidEnclosingCenteredEllipsoidRadial(lambda_mu * torch.eye(d))
        super().__init__(d, ellipsoid_radial, trim_radial, softmin=softmin) 