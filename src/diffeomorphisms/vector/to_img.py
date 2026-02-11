from src.diffeomorphisms.vector import VectorDiffeomorphism

class ToImgVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, in_channels, height, width):
        super().__init__(in_channels * height* width)
        self.C = in_channels
        self.H = height
        self.W = width

    def forward(self, x):
        """
        
        :param x: N x (C*H*W)
        :return: N x C x H x W
        """
        N = x.shape[0]
        return x.reshape(N, self.C, self.H, self.W)
    
    def inverse(self, y):
        """
        
        :param y: N x C x H x W
        :return: N x (C*H*W)
        """
        N = y.shape[0]
        return y.reshape(N, self.C * self.H * self.W)
    
    def differential_forward(self, x, X):
        """
        
        :param x: N x (C*H*W)
        :param X: N x (C*H*W)
        :return: N x C x H x W
        """
        N = X.shape[0]
        return X.reshape(N, self.C, self.H, self.W)
    
    def differential_inverse(self, y, Y):
        """
        
        :param y: N x C x H x W
        :param Y: N x C x H x W
        :return: N x (C*H*W)
        """
        N = Y.shape[0]
        return Y.reshape(N, self.C * self.H * self.W)