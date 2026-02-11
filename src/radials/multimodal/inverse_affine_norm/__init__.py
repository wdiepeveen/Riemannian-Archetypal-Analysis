from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.inverse_affine_norm.pre_bias_diagonal import InversePreBiasDiagonalNormRadial

class MultiInverseAffineNormRadial(MultiModalRadial):
    def __init__(self, d, radials):
        super().__init__(d, radials)

    def mode_mean(self, idx):
        """
        Return the mean of the idx-th mode of the multimodal radial.
        :param idx: int
        :return: tensor of shape (d,)
        """
        return self.radials[idx].mode_mean
    
    def mode_variance(self, idx):
        """
        Return the variance of the idx-th mode of the multimodal radial.
        :param idx: int
        :return: tensor of shape (d, d)
        """
        return self.radials[idx].mode_variance