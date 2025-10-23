import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class RiverVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, beta, eta) -> None:
        super().__init__(2)
        self.beta = beta
        self.eta = eta

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y = x.clone()
        y[:, 0] = y[:, 0] - self.beta * torch.sin(y[:, 1])
        y[:, 1] = torch.sinh(self.eta * y[:, 1])
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 2), where N is the batch size.
        :return: Inverted tensor of shape (N, 2).
        """
        x = y.clone()
        x[:, 0] = y[:, 0] + self.beta * torch.sin(torch.arcsinh(y[:, 1]) / self.eta)
        x[:, 1] = torch.arcsinh(y[:, 1]) / self.eta
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        dx = X.clone()
        dx0 = X[:, 0] - self.beta * torch.cos(x[:, 1]) * X[:, 1]
        dx1 = self.eta * torch.cosh(self.eta * x[:, 1]) * X[:, 1]
        dx[:, 0] = dx0
        dx[:, 1] = dx1
        return dx


    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 2), inputs.
        :param Y: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        x1 = torch.arcsinh(y[:, 1]) / self.eta  # (N,)
        d_x1_d_y1 = 1.0 / (self.eta * torch.sqrt(1 + y[:, 1] ** 2))  # (N,)
        dx = Y.clone()
        dx[:, 1] = d_x1_d_y1 * Y[:, 1]
        dx[:, 0] = Y[:, 0] + self.beta * torch.cos(x1) * d_x1_d_y1 * Y[:, 1]
        return dx