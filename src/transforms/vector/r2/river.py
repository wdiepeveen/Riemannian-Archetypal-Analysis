import torch

from src.transforms.vector import VectorTransform


class RiverVectorTransform(VectorTransform):
    def __init__(self, shear, offset) -> None:
        super().__init__(2)
        self.a = shear
        self.z = offset

        rot = torch.tensor([[1., -1.], [1., 1.]]) / (2 ** 0.5)
        inv_rot = torch.tensor([[1., 1.], [-1., 1.]]) / (2 ** 0.5)

        self.register_buffer("rot", rot)
        self.register_buffer("inv_rot", inv_rot)

    def inverse(self, x, context=None):
        u = x @ self.rot.T
        y = u.clone()
        y[:, 0] = u[:, 0] + (self.a * u[:, 1]).sin() - self.z
        # y[:, 0] = u[:, 0] + self.a * u[:, 1] ** 3 - self.z
        return y, x.new_zeros(x.shape[0])

    def forward(self, y, context=None):
        u = y.clone()
        u[:, 0] = y[:, 0] - (self.a * y[:, 1]).sin() + self.z
        # u[:, 0] = y[:, 0] - self.a * y[:, 1] ** 3 + self.z
        x = u @ self.inv_rot.T
        return x, y.new_zeros(y.shape[0])