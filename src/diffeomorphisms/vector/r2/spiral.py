import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class SpiralVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, beta) -> None:
        super().__init__(2)
        self.beta = beta

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        x1, x2 = x[:, 0], x[:, 1]
        R = torch.sqrt(x1**2 + x2**2)
        r = R / self.beta
        phi = torch.atan2(x2, x1)
        theta = (phi - r) % (2 * torch.pi)
        return torch.stack((r, theta), dim=1)

    def inverse(self, p):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param p: Tensor of shape (N, 2), where N is the batch size, in polar coordinates.
        :return: Inverted tensor of shape (N, 2).
        """
        r, theta = p[:, 0], p[:, 1] 
        x = self.beta * r[:,None] * torch.stack([torch.cos(r + theta), torch.sin(r + theta)], dim=-1)
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        x1, x2 = x[:, 0], x[:, 1]
        X1, X2 = X[:, 0], X[:, 1]
        R = torch.sqrt(x1**2 + x2**2)
        r = R / self.beta

        # Avoid division by 0 for R
        eps = 1e-12
        R_safe = R + eps

        # Partial derivatives for r and theta with respect to x1, x2
        dr_dx1 = x1 / (self.beta * R_safe)
        dr_dx2 = x2 / (self.beta * R_safe)

        dphi_dx1 = -x2 / (R_safe**2)
        dphi_dx2 = x1 / (R_safe**2)

        dtheta_dx1 = dphi_dx1 - dr_dx1
        dtheta_dx2 = dphi_dx2 - dr_dx2

        # Apply Jacobian to X
        dr = dr_dx1 * X1 + dr_dx2 * X2
        dtheta = dtheta_dx1 * X1 + dtheta_dx2 * X2

        return torch.stack((dr, dtheta), dim=1)


    def differential_inverse(self, p, P):
        """
        Computes the differential of the inverse transformation.
        :param p: Tensor of shape (N, 2), inputs.
        :param P: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        r, theta = p[:, 0], p[:, 1]
        pr, ptheta = P[:, 0], P[:, 1]
        phi = r + theta

        # Jacobian components
        dx1_dr = self.beta * (torch.cos(phi) - r * torch.sin(phi))
        dx1_dtheta = -self.beta * r * torch.sin(phi)
        dx2_dr = self.beta * (torch.sin(phi) + r * torch.cos(phi))
        dx2_dtheta = self.beta * r * torch.cos(phi)

        # Apply Jacobian to P
        dx1 = dx1_dr * pr + dx1_dtheta * ptheta
        dx2 = dx2_dr * pr + dx2_dtheta * ptheta

        return torch.stack((dx1, dx2), dim=1)