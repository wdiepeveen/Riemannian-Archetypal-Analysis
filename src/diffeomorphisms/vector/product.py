import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class ProductVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, diffeomorphisms):
        super().__init__(sum(d.d for d in diffeomorphisms))

        self.diffeomorphisms = diffeomorphisms

    def forward(self, x):
        """
        Apply the product diffeomorphism to x.
        
        :param x: N x d
        :return: N x d
        """
        outputs = []
        start = 0
        for diffeo in self.diffeomorphisms:
            end = start + diffeo.d
            outputs.append(diffeo.forward(x[:, start:end]))
            start = end
        return torch.cat(outputs, dim=-1)
    
    def inverse(self, y):
        """
        Apply the inverse of the product diffeomorphism to y.
        
        :param y: N x d
        :return: N x d
        """
        outputs = []
        start = 0
        for diffeo in self.diffeomorphisms:
            end = start + diffeo.d
            outputs.append(diffeo.inverse(y[:, start:end]))
            start = end
        return torch.cat(outputs, dim=-1)
    
    def differential_forward(self, x, X):
        """
        Apply the differential of the product diffeomorphism to x and X.
        
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        outputs = []
        start = 0
        for diffeo in self.diffeomorphisms:
            end = start + diffeo.d
            outputs.append(diffeo.differential_forward(x[:, start:end], X[:, start:end]))
            start = end
        return torch.cat(outputs, dim=-1)
    
    def differential_inverse(self, y, Y):
        """
        Apply the differential of the inverse of the product diffeomorphism to y and Y.
        
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        outputs = []
        start = 0
        for diffeo in self.diffeomorphisms:
            end = start + diffeo.d
            outputs.append(diffeo.differential_inverse(y[:, start:end], Y[:, start:end]))
            start = end
        return torch.cat(outputs, dim=-1)