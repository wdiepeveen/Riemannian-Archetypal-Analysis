import torch

from src.riemannian_neural_networks.archetypal_mappings import RiemannianArchetypalMapping
from src.riemannian_neural_networks.archetypal_mappings.relaxed import RelaxedRiemannianArchetypalMapping   

class ExactRiemannianArchetypalMapping(RiemannianArchetypalMapping):
    def __init__(self, euclidean_pullback_manifold, archetypes, init_euclidean_pullback_manifold=None, max_iter=500, tol=1e-6, accelerated=True, line_search=True, ls_beta=0.5, ls_c=1e-4, ls_max_iter=20):
        super().__init__(euclidean_pullback_manifold, archetypes, max_iter=max_iter, tol=tol, accelerated=accelerated, line_search=line_search, ls_beta=ls_beta, ls_c=ls_c, ls_max_iter=ls_max_iter)
        self.phi = self.manifold.phi
        self.phi_m = self.phi(self.m) # (r, d)
        self.init_manifold = init_euclidean_pullback_manifold

        # step size
        G = self.phi_m.reshape(self.r, self.d) @ self.phi_m.reshape(self.r, self.d).T
        evals = torch.linalg.eigvalsh(G)
        L = 2.0 * torch.clamp(evals[-1], min=1e-12)
        self.initial_step_size = 1.0 / L
    
    def archetype_weights_init(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        if self.init_manifold is not None:
            relaxed_ram = RelaxedRiemannianArchetypalMapping(self.init_manifold, self.m, max_iter=self.max_iter, tol=self.tol, accelerated=self.accelerated) 
        else:
            relaxed_ram = RelaxedRiemannianArchetypalMapping(self.manifold, self.m, max_iter=self.max_iter, tol=self.tol, accelerated=self.accelerated)

        return relaxed_ram.archetype_weights_init(x)
    
    # def archetype_weights_init(self, x):
    #     """
    #     :param x: N x [input_dim] tensor
    #     :return: N x r tensor of archetypal coefficients
    #     """
    #     N = x.shape[0]
    #     phi_x = self.phi(x)  # (N, d)
    #     pairwise_distances = torch.cdist(phi_x.reshape(N, self.d), self.phi_m.reshape(self.r, self.d))  # (N, r)
    #     return torch.softmax(-pairwise_distances, dim=-1)
    
    def objective(self, w, x):
        """
        Computes the objective function value for the given archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: scalar tensor representing the objective function value
        """
        N = x.shape[0]
        z = (w @ self.phi_m.reshape(self.r, self.d)).reshape(N, *x.shape[1:])
        x_hat = self.phi.inverse(z)
        loss = ((x - x_hat) ** 2).sum()
        return loss
    
    def gradient(self, w, x):
        """
        Computes the gradient of the objective function with respect to the archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N x r tensor of gradients
        """
        with torch.enable_grad():
            w_ = w.detach().clone().requires_grad_(True)
            obj = (lambda w_local: self.objective(w_local, x))(w_)
            grad, = torch.autograd.grad(obj, w_, create_graph=True)
            return grad
    