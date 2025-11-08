import torch

from src.manifolds.hyperbolic.vector import VectorHyperbolic

class StandardVectorHyperbolic(VectorHyperbolic):
    """ Base class describing Hyperbolic space of dimension d modelled as the Poincaré ball """

    def __init__(self, d):
        super().__init__(d)
    
    def lambda_factor(self, x):
        norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        return 2 / (1 - norm_x_sq)

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        lambda_x = self.lambda_factor(x)   # N x 1
        # Broadcast for batch dimensions
        Xb = X.unsqueeze(2)                # N x M x 1 x d
        Yb = Y.unsqueeze(1)                # N x 1 x L x d
        inner_product = torch.sum(Xb * Yb, dim=-1)  # N x M x L
        return (lambda_x.unsqueeze(-1).unsqueeze(-1) ** 2) * inner_product

    def norm(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M
        """
        lambda_x = self.lambda_factor(x)   # N x 1
        norm_euc = torch.norm(X, dim=-1)   # N x M
        return lambda_x.unsqueeze(-1) * norm_euc

    def barycentre(self, x, tol=1e-5, max_iter=100, step_size=1.0, red_coef=0.5):
        """

        :param x: N x d
        :return: d
        """
        mu = torch.mean(x, dim=0, keepdim=True)  # d
        for _ in range(max_iter):
            # Compute average logarithmic directions
            v = self.log(mu, x.unsqueeze(0)).mean(dim=1)  # (1 x N x d) --> (1 x d)
            norm_v = torch.norm(v, dim=-1, keepdim=True)
            if torch.max(norm_v) < tol:
                break
            mu = self.exp(mu, step_size * v)
        return mu.squeeze(0)

    def mob_add(self, x, y):
        # Möbius addition for Poincaré ball
        norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm_y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * xy_dot + norm_y_sq) * x + (1 - norm_x_sq) * y
        denominator = 1 + 2 * xy_dot + norm_x_sq * norm_y_sq
        return numerator / denominator.clamp(min=1e-7)

    def geodesic(self, x, y, t):
        """

        :param x: N x M x d
        :param y: N x L x d
        :param t: K or N x M x L x K
        :return: N x M x L x K x d
        """
        dxy = self.distance(x, y)  # N x M x L
        direction = self.log(x, y)  # N x M x L x d
        
        if t.dim() == 1:
            t_shape = list(dxy.shape) + [len(t)]
            t_mat = t.view(1, 1, 1, -1).expand(*t_shape)
        else:
            t_mat = t
        vec = direction.unsqueeze(-2) * t_mat.unsqueeze(-1)  # N x M x L x K x d
        return self.exp(x.unsqueeze(2).unsqueeze(-2), vec)

    def log(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L x d
        """
        norm_diff = torch.norm(y - x.unsqueeze(2), dim=-1, keepdim=True)  # N x M x L x 1
        lambda_x = self.lambda_factor(x).unsqueeze(2)                     # N x M x 1 x 1
        diff = y - x.unsqueeze(2)                                         # N x M x L x d

        # Hyperbolic distance
        dist = self.distance(x, y).unsqueeze(-1)                          # N x M x L x 1

        return (2 / lambda_x) * torch.atanh(norm_diff / dist.clamp(min=1e-7)) * (diff / (norm_diff.clamp(min=1e-7)))

    def exp(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        norm_X = torch.norm(X, dim=-1, keepdim=True)  # N x M x 1
        lambda_x = self.lambda_factor(x).unsqueeze(1) # N x 1 x 1
        factor = torch.tanh(lambda_x * norm_X / 2) * X / (norm_X.clamp(min=1e-7))
        return self.mob_add(x.unsqueeze(1), factor)

    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # N x M x 1
        norm_y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)  # N x 1 x L
        diff = y - x.unsqueeze(2)        # N x M x L x d
        norm_diff_sq = torch.sum(diff ** 2, dim=-1)  # N x M x L

        num = 2 * norm_diff_sq
        denom = (1 - norm_x_sq) * (1 - norm_y_sq)
        arg = 1 + num / denom.clamp(min=1e-7)
        return torch.acosh(arg.clamp(min=1.0))

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x d
        :param X: N x M x K x d
        :param y: N x L x d
        :return: N x M x L x K x d
        """
        log_xy = self.log(x, y)                   # N x M x L x d
        norm_log_xy = torch.norm(log_xy, dim=-1, keepdim=True) # N x M x L x 1
        u = log_xy / (norm_log_xy.clamp(min=1e-7))             # N x M x L x d
        X = X.unsqueeze(2)                                     # N x M x 1 x K x d

        dot_uX = torch.sum(u.unsqueeze(-2) * X, dim=-1, keepdim=True) # N x M x L x K x 1
        X_parallel = X - dot_uX * u.unsqueeze(-2)                    # N x M x L x K x d

        theta = self.distance(x, y).unsqueeze(-1).unsqueeze(-1)      # N x M x L x 1 x 1

        return X_parallel * torch.cos(theta) + dot_uX * u.unsqueeze(-2) * torch.cos(theta) + torch.cross(u.unsqueeze(-2), X_parallel, dim=-1) * torch.sin(theta)
    