from src.diffeomorphisms.image import ImageDiffeomorphism

class ToVecImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width):
        super().__init__(in_channels, height, width)

    def forward(self, x):
        """
        
        :param x: N x C x H x W
        :return: N x (C*H*W)
        """
        N, C, H, W = x.shape
        return x.reshape(N, C * H * W)
    
    def inverse(self, y):
        """
        
        :param y: N x (C*H*W)
        :return: N x C x H x W
        """
        N = y.shape[0]
        return y.reshape(N, self.C, self.H, self.W)
    
    def differential_forward(self, x, X):
        """
        
        :param x: N x C x H x W
        :param X: N x C x H x W
        :return: N x (C*H*W)
        """
        N, C, H, W = X.shape
        return X.reshape(N, C * H * W)
    
    def differential_inverse(self, y, Y):
        """
        
        :param y: N x (C*H*W)
        :param Y: N x (C*H*W)
        :return: N x C x H x W
        """
        N = Y.shape[0]
        return Y.reshape(N, self.C, self.H, self.W)