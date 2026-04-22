import torch

from src.riemannian_neural_networks.archetypal_mappings import RiemannianArchetypalMapping
from src.riemannian_neural_networks.archetypal_mappings.relaxed import RelaxedRiemannianArchetypalMapping   

class ExactRiemannianArchetypalMapping(RiemannianArchetypalMapping):
    def __init__(self, euclidean_pullback_manifold, archetypes):
        super().__init__(euclidean_pullback_manifold, archetypes)
        self.phi = self.manifold.phi
        self.phi_m = self.phi(self.m) # (r, d)
    
    def archetype_weights_init(self, x, euclidean_pullback_manifold=None):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        if euclidean_pullback_manifold is None:
            euclidean_pullback_manifold = self.manifold
        relaxed_ram = RelaxedRiemannianArchetypalMapping(euclidean_pullback_manifold, self.m)
        return relaxed_ram.archetype_weights_init(x)
    
    def gradient(self, w, x):
        """
        Computes the gradient of the objective function with respect to the archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N x r tensor of gradients
        """
        w_ = w.detach().clone().requires_grad_(True)
        obj = self.objective(w_, x)
        grad, = torch.autograd.grad(obj, w_, create_graph=False)
        return grad


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
        return ((x - x_hat) ** 2).sum()
    