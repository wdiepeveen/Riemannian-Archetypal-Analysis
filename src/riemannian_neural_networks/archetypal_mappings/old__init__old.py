import torch

from src.riemannian_neural_networks import RiemannianNeuralNetwork

class RiemannianArchetypalMapping(RiemannianNeuralNetwork):
    def __init__(self, manifold, archetypes, max_iter=200, tol=1e-6, accelerated=True, line_search=False):
        super().__init__(manifold)
        self.m = archetypes
        self.r = archetypes.shape[0]

        self.max_iter = max_iter
        self.tol = tol
        self.accelerated = accelerated
        self.line_search = line_search
        self.step_size = None # will be set by subclass, as it may depend on the specific objective function being optimized

    def forward(self, x, return_weights=False):
        """
        Computes T(x) := argmin_{y} \sum_{j=1}^r g_j^* d(y, m^j)^2, where g^* are the archetype weights implemented by subclass
        :param x: N x [input_dim] tensor
        """
        weights = self.archetype_weights(x)
        barycentre = self.manifold.barycentre(self.m, weights=weights.T)
        if return_weights:
            return barycentre, weights
        return barycentre
    
    def archetype_weights(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        w = self.archetype_weights_init(x)
        N = x.shape[0]
        if self.accelerated:
            v = w.clone()
            t = torch.ones(N, 1, device=x.device, dtype=x.dtype)

            for _ in range(self.max_iter):
                grad_v = self.gradient(v, x)
                w_next = self._project_simplex(v - self.step_size * grad_v)

                diff = torch.max(torch.abs(w_next - w))
                if diff < self.tol:
                    w = w_next
                    break

                t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
                v = w_next + ((t - 1.0) / t_next) * (w_next - w)

                w = w_next
                t = t_next
        else:
            for _ in range(self.max_iter):
                grad_w = self.gradient(w, x)
                w_next = self._project_simplex(w - self.step_size * grad_w)

                diff = torch.max(torch.abs(w_next - w))
                w = w_next
                if diff < self.tol:
                    break

        return w
    
    def archetype_weights_init(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        raise NotImplementedError("Subclasses must implement archetype_weights_init method")
    
    def objective(self, w, x):
        """
        Computes the objective function value for the given archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: scalar tensor representing the objective function value
        """
        raise NotImplementedError("Subclasses must implement objective method")
    
    def gradient(self, w, x):
        """
        Computes the gradient of the objective function with respect to the archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N x r tensor of gradients
        """
        raise NotImplementedError("Subclasses must implement gradient method")
    
    @staticmethod
    def _project_simplex(v):
        """
        Projects each row of v onto the probability simplex.
        :param v: N x r tensor
        :return: N x r tensor, projection of v onto the probability simplex
        """
        u, _ = torch.sort(v, dim=-1, descending=True)
        cssv = torch.cumsum(u, dim=-1) - 1.0
        j = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / j > 0
        rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        theta = cssv.gather(-1, rho - 1) / rho
        return torch.clamp(v - theta, min=0.0)
    