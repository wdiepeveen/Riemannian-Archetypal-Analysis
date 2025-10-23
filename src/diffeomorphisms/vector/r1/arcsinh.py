import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class ArcsinhVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, offset=1.) -> None:
        super().__init__(1)
        self.offset = offset

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 1), where N is the batch size.
        :return: Transformed tensor of shape (N, 1).
        """
        return x.arcsinh() - self.offset

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 1), where N is the batch size.
        :return: Inverted tensor of shape (N, 1).
        """
        return (y + self.offset).sinh()

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 1), inputs.
        :param X: Tensor of shape (N, 1), differentials to transform.
        :return: Transformed differential tensor of shape (N, 1).
        """
        return X / (x.pow(2) + 1).sqrt()

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 1), inputs.
        :param Y: Tensor of shape (N, 1), differentials to invert.
        :return: Inverted differential tensor of shape (N, 1).
        """
        return (y + self.offset).cosh() * Y
        