import torch

from src.riemannian_neural_networks.archetypal_mappings import RiemannianArchetypalMapping
from src.riemannian_neural_networks.archetypal_mappings.relaxed import RelaxedRiemannianArchetypalMapping   

class ExactRiemannianArchetypalMapping(RiemannianArchetypalMapping):
    def __init__(self, euclidean_pullback_manifold, archetypes, init_euclidean_pullback_manifold=None, max_iter=200, tol=1e-6, accelerated=True, line_search=True, ls_beta=0.5, ls_c=1e-4, ls_max_iter=20):
        super().__init__(euclidean_pullback_manifold, archetypes, max_iter=max_iter, tol=tol, accelerated=accelerated, line_search=line_search, ls_beta=ls_beta, ls_c=ls_c, ls_max_iter=ls_max_iter)
        self.phi = self.manifold.phi
        self.phi_m = self.phi(self.m) # (r, d)
        if init_euclidean_pullback_manifold is not None:
            self.relaxed_ram = RelaxedRiemannianArchetypalMapping(init_euclidean_pullback_manifold, self.m, max_iter=self.max_iter, tol=self.tol, accelerated=self.accelerated) 
        else:
            self.relaxed_ram = RelaxedRiemannianArchetypalMapping(self.manifold, self.m, max_iter=self.max_iter, tol=self.tol, accelerated=self.accelerated)

        self.step_size = self.relaxed_ram.step_size * 1e-2
    
    def archetype_weights_init(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        return self.relaxed_ram.archetype_weights_init(x)
    
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
    