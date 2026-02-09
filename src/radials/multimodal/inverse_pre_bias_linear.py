from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.inverse_pre_bias_linear import InversePreBiasLinearRadial

class MultiInversePreBiasLinearRadial(MultiModalRadial):
    def __init__(self, d, num_radials, centered=True):
        super().__init__(d, [InversePreBiasLinearRadial(d, centered=centered) for _ in range(num_radials)])


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