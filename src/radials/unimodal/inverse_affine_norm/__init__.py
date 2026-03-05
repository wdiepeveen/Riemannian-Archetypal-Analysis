from src.radials.unimodal import UniModalRadial

class InverseAffineNormRadial(UniModalRadial):
    def __init__(self, d, sigma=1e-3):
        super().__init__(d)
        self.sigma = sigma

    def forward(self, theta):
        return 1/(self.affine(theta).norm(dim=-1)**2 + self.sigma **2).sqrt()
    
    def affine(self, theta):
        """
        Compute the affine transformation of theta.
        :param theta: N x d
        :return: N x r
        """
        raise NotImplementedError
    
    @property
    def mode_mean(self):
        """
        Return the mean of the mode of the radial.
        """
        raise NotImplementedError
        
    @property
    def mode_variance(self):
        """
        Return the variance of the mode of the radial.
        """
        raise NotImplementedError