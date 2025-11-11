import torch

from src.manifolds.hyperbolic.vector import VectorHyperbolic

class StandardVectorHyperbolic(VectorHyperbolic):
    """ Base class describing Hyperbolic space of dimension d modelled as the Poincaré ball """

    def __init__(self, d):
        super().__init__(d)
    
    def lambda_factor(self, x):
        """
        :param x: N x d
        :return: N
        """
        norm_x_sq = torch.sum(x ** 2, dim=-1)
        return 2 / (1 - norm_x_sq)
    
    def mob_add(self, x, y):
        """
        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L x d
        """
        # Möbius addition for Poincaré ball
        norm_x_sq = torch.sum(x ** 2, dim=-1) # N x M
        norm_y_sq = torch.sum(y ** 2, dim=-1) # N x L
        xy_dot = torch.sum(x.unsqueeze(2) * y.unsqueeze(1), dim=-1) # N x M x L
        numerator = (1 + 2 * xy_dot + norm_y_sq.unsqueeze(1)).unsqueeze(-1) * x.unsqueeze(2) + (1 - norm_x_sq.unsqueeze(2).unsqueeze(-1)) * y.unsqueeze(1)
        denominator = 1 + 2 * xy_dot + norm_x_sq.unsqueeze(2) * norm_y_sq.unsqueeze(1)
        return numerator / denominator.clamp(min=1e-7).unsqueeze(-1)
    
    def gyration(self, x, y, X):
        """
        :param x: N x M x d
        :param y: N x L x d
        :param X: N x M x K x d
        :return: N x M x L x K x d
        """
        N,M,L,K = x.shape[0], x.shape[1], y.shape[1], X.shape[2]
        xy = self.mob_add(x, y)  # N x M x L x d
        yX = self.mob_add(y, X.reshape(N, -1, self.d)).reshape(N, L, M, K, self.d).permute(0, 2, 1, 3, 4)  # N x M x L x K x d
        x_yX = self.mob_add(x.reshape(N * M, 1, self.d), yX.reshape(N * M, -1, self.d)).reshape(N, M, L, K, self.d)  # N x M x L x K x d
        return self.mob_add(-xy.reshape(N * M * L, 1, self.d), x_yX.reshape(N * M * L, K, self.d)).reshape(N, M, L, K, self.d)  # N x M x L x K x d
        

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        lambda_x = self.lambda_factor(x)   # N
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
        lambda_x = self.lambda_factor(x)   # N
        norm_euc = torch.norm(X, dim=-1)   # N x M
        return lambda_x.unsqueeze(-1) * norm_euc

    def barycentre(self, x, tol=1e-5, max_iter=100, step_size=1.0, red_coef=0.5):
        """

        :param x: N x d
        :return: d
        """
        mu = torch.mean(x, dim=0, keepdim=True)  # 1 x d
        for _ in range(max_iter):
            # Compute average logarithmic directions
            v = self.log(mu.unsqueeze(0), x.unsqueeze(0)).mean(dim=2).squeeze(0)  # (1 x 1 x N x d) --> (1 x d)
            norm_v = torch.norm(v, dim=-1, keepdim=True)
            if torch.max(norm_v) < tol:
                break
            mu = self.exp(mu, step_size * v.unsqueeze(0)).squeeze(0)
        return mu.squeeze(0)

    def geodesic(self, x, y, t):
        """

        :param x: N x M x d
        :param y: N x L x d
        :param t: K or N x M x L x K
        :return: N x M x L x K x d
        """
        N, M, L, K = x.shape[0], x.shape[1], y.shape[1], t.shape[-1] if t.dim() > 1 else len(t)
        log = self.log(x, y)  # N x M x L x d
        
        if t.dim() == 1:
            t_mat = t.view(1, 1, 1, -1).repeat(N, M, L, 1)  # N x M x L x K
        else:
            t_mat = t
        tlog = log.unsqueeze(-2) * t_mat.unsqueeze(-1)  # N x M x L x K x d
        return self.exp(x.reshape(N * M, self.d), tlog.reshape(N * M, L * K, self.d)).reshape(N, M, L, K, self.d)

    def log(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L x d
        """
        lambda_x = self.lambda_factor(x.reshape(-1,self.d)).reshape(x.shape[:-1]) # N x M
        diff = self.mob_add(-x, y) # N x M x L x d
        norm_diff = torch.norm(diff, dim=-1) # N x M x L
        return (2 / lambda_x.unsqueeze(2).unsqueeze(-1)) * torch.atanh(norm_diff).unsqueeze(-1) * (diff / (norm_diff.clamp(min=1e-7).unsqueeze(-1)))

    def exp(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        norm_X = torch.norm(X, dim=-1)  # N x M 
        lambda_x = self.lambda_factor(x) # N
        factor = torch.tanh(lambda_x.unsqueeze(-1) * norm_X / 2).unsqueeze(-1) * X / (norm_X.clamp(min=1e-7).unsqueeze(-1))
        return self.mob_add(x.unsqueeze(1), factor).squeeze(1)

    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        norm_x_sq = torch.sum(x ** 2, dim=-1)  # N x M
        norm_y_sq = torch.sum(y ** 2, dim=-1)  # N x L
        diff = y.unsqueeze(1) - x.unsqueeze(2)        # N x M x L x d
        norm_diff_sq = torch.sum(diff ** 2, dim=-1)  # N x M x L

        num = 2 * norm_diff_sq
        denom = (1 - norm_x_sq).unsqueeze(2) * (1 - norm_y_sq).unsqueeze(1)  # N x M x L
        arg = 1 + num / denom.clamp(min=1e-7)
        return torch.acosh(arg.clamp(min=1.0))

    def parallel_transport(self, x, X, y): # TODO verify
        """

        :param x: N x M x d
        :param X: N x M x K x d
        :param y: N x L x d
        :return: N x M x L x K x d
        """
        gyr_xy_X = self.gyration(x, y, X)  # N x M x L x K x d
        lambda_x = self.lambda_factor(x.reshape(-1,self.d)).reshape(x.shape[:-1])  # N x M
        lambda_y = self.lambda_factor(y.reshape(-1,self.d)).reshape(y.shape[:-1])  # N x L
        scale = (lambda_x.unsqueeze(2) / lambda_y.unsqueeze(1))  # N x M x L
        return scale.unsqueeze(-1).unsqueeze(-1) * gyr_xy_X