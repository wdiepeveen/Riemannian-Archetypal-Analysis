import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class TreeVectorDiffeomorphism(VectorDiffeomorphism):
    """ Diffeomorphism that maps the tetrahedron 
        [[1.0, 1.0, 1.0], 
        [1.0, -1.0, -1.0], 
        [-1.0, 1.0, -1.0], 
        [-1.0, -1.0, 1.0]] 
    in R^3 to 
        [[ 1.0, -1.0,  0.0],
        [ 1.0,  1.0,  0.0],
        [-1.0,  1.0,  0.0],
        [-1.0, -1.0,  0.0]]
    """
    def __init__(self):
        super().__init__(3)

    def forward(self, x):
        N = x.shape[0]
        # non-linearly rotate around the x-axis
        angle = x[:, 0] * (3.141592653589793 / 4)  # angle varies from -pi/4 to pi/4
        R = torch.zeros((N, 3, 3))
        R[:, 0, 0] = 1
        R[:, 1, 1] = torch.cos(angle)
        R[:, 1, 2] = -torch.sin(angle)
        R[:, 2, 1] = torch.sin(angle)
        R[:, 2, 2] = torch.cos(angle)

        out = torch.einsum('nij,nj->ni', R, x)

        # apply global rotation
        R_global = torch.tensor([[ 1.,  0.,  0.],
                                  [ 0.,  0., -1.],
                                  [ 0.,  1.,  0.]])
        out = torch.einsum('ij,nj->ni', R_global, out)

        # rescale y-axis to preserve distances
        out[:, 1] = out[:, 1] / (2 ** 0.5)

        return out

    def inverse(self, y):
        N = y.shape[0]
        # Undo the rescaling of the y-axis:
        out = y.clone()
        out[:, 1] = out[:, 1] * (2 ** 0.5)
        # Undo the global rotation (transpose = inverse for rotation)
        R_global = torch.tensor([[1., 0., 0.],
                                 [0., 0., -1.],
                                 [0., 1., 0.]], dtype=y.dtype, device=y.device)
        out = torch.einsum('ji,nj->ni', R_global, out)  # transpose global rotation
        # Undo the local x-dependent rotation
        angle = out[:, 0] * (3.141592653589793 / 4)
        R = torch.zeros((N, 3, 3), dtype=y.dtype, device=y.device)
        R[:, 0, 0] = 1
        R[:, 1, 1] = torch.cos(angle)
        R[:, 1, 2] = torch.sin(angle)   # Notice: here we use +sin, inverse rotation!
        R[:, 2, 1] = -torch.sin(angle)
        R[:, 2, 2] = torch.cos(angle)
        x = torch.einsum('nij,nj->ni', R, out)
        return x

    def differential_forward(self, x, X):
        # X is tangent vector (N, 3); differential is the linear map applied at x
        N = x.shape[0]
        angle = x[:, 0] * (3.141592653589793 / 4)
        R = torch.zeros((N, 3, 3), dtype=x.dtype, device=x.device)
        R[:, 0, 0] = 1
        R[:, 1, 1] = torch.cos(angle)
        R[:, 1, 2] = -torch.sin(angle)
        R[:, 2, 1] = torch.sin(angle)
        R[:, 2, 2] = torch.cos(angle)
        # Apply batch rotation to tangent
        Xf = torch.einsum('nij,nj->ni', R, X)
        # Global rotation
        R_global = torch.tensor([[1., 0., 0.],
                                 [0., 0., -1.],
                                 [0., 1., 0.]], dtype=x.dtype, device=x.device)
        Xf = torch.einsum('ij,nj->ni', R_global, Xf)
        # Rescale y
        Xf[:, 1] = Xf[:, 1] / (2 ** 0.5)
        return Xf

    def differential_inverse(self, y, Y):
        # Y is tangent vector (N, 3)
        N = y.shape[0]
        # Undo rescale
        Y_inv = Y.clone()
        Y_inv[:, 1] = Y_inv[:, 1] * (2 ** 0.5)
        # Undo global rotation (transpose)
        R_global = torch.tensor([[1., 0., 0.],
                                 [0., 0., -1.],
                                 [0., 1., 0.]], dtype=y.dtype, device=y.device)
        Y_inv = torch.einsum('ji,nj->ni', R_global, Y_inv)
        # Undo local rotation (transpose, i.e. -angle)
        angle = Y_inv[:, 0] * (3.141592653589793 / 4)
        R = torch.zeros((N, 3, 3), dtype=y.dtype, device=y.device)
        R[:, 0, 0] = 1
        R[:, 1, 1] = torch.cos(angle)
        R[:, 1, 2] = torch.sin(angle)
        R[:, 2, 1] = -torch.sin(angle)
        R[:, 2, 2] = torch.cos(angle)
        X = torch.einsum('nij,nj->ni', R, Y_inv)
        return X