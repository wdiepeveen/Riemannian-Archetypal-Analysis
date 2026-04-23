import torch

from src.riemannian_neural_networks.archetypal_mappings import RiemannianArchetypalMapping

class RelaxedRiemannianArchetypalMapping(RiemannianArchetypalMapping):
    def __init__(self, euclidean_pullback_manifold, archetypes, max_iter=500, tol=1e-6, accelerated=True):
        super().__init__(euclidean_pullback_manifold, archetypes, max_iter=max_iter, tol=tol, accelerated=accelerated)
        self.phi = self.manifold.phi
        self.phi_m = self.phi(self.m) # (r, d)

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
        N = x.shape[0]
        phi_x = self.phi(x)  # (N, d)
        pairwise_distances = torch.cdist(phi_x.reshape(N, self.d), self.phi_m.reshape(self.r, self.d))  # (N, r)
        return torch.softmax(-pairwise_distances, dim=-1)
    
    def gradient(self, w, x):
        """
        Computes the gradient of the objective function with respect to the archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N x r tensor of gradients
        """
        N = x.shape[0]
        phi_x = self.phi(x).reshape(N, self.d)  # (N, d)
        phi_barycentre = w @ self.phi_m.reshape(self.r, self.d)  # (N, d)
        diff = phi_barycentre - phi_x  # (N, d)
        grad = 2.0 * diff @ self.phi_m.reshape(self.r, self.d).T  # (N, r)
        return grad