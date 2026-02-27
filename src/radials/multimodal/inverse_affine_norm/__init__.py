from src.radials.multimodal import MultiModalRadial

class MultiInverseAffineNormRadial(MultiModalRadial):
    def __init__(self, d, radials):
        super().__init__(d, radials)
        
    def mode_variance(self, idx):
        """
        Return the variance of the idx-th mode of the multimodal radial.
        :param idx: int
        :return: tensor of shape (d, d)
        """
        return self.radials[idx].mode_variance