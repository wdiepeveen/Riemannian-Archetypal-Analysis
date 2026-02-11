from src.distributions.star_gaussians import StarGaussianDistribution
from src.radials.multimodal.inverse_affine_norm.pre_bias_diagonal import MultiInversePreBiasDiagonalNormRadial

class MultiInversePreBiasDiagonalStarNormGaussianDistribution(StarGaussianDistribution):
    def __init__(self, d, num_radials, centered=True):
        super().__init__(MultiInversePreBiasDiagonalNormRadial(d, num_radials, centered=centered))
        
    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal star Gaussian distribution.
        :param idx: int
        :return: StarGaussianDistribution with the same radial as the idx-th mode of the multimodal radial
        """
        return StarGaussianDistribution(self.radial.extract_mode(idx))