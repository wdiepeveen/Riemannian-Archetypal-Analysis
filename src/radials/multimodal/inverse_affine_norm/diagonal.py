from src.radials.multimodal.inverse_affine_norm import MultiInverseAffineNormRadial
from src.radials.unimodal.inverse_affine_norm.diagonal import InverseDiagonalNormRadial

class MultiInverseDiagonalNormRadial(MultiInverseAffineNormRadial):
    def __init__(self, d, num_radials, sigma=1e-6):
        super().__init__(d, [InverseDiagonalNormRadial(d, sigma=sigma) for _ in range(num_radials)])
    
    def mode_variance(self, idx):
        """
        Return the variance of the idx-th mode of the multimodal radial.
        :param idx: int
        :return: tensor of shape (d, d)
        """
        return self.radials[idx].mode_variance