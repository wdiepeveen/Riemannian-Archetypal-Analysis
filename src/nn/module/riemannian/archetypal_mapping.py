import torch

from src.dimension_reduction.archetypal_analysis import ArchetypalAnalysisSolver

class RiemannianArchetypalMapping(torch.nn.Module):
    def __init__(self, Omega_manifold, archetypes):
        super().__init__()
        self.Omega_manifold = Omega_manifold
        self.archetypes = archetypes
        self.r = archetypes.shape[0]

        with torch.no_grad():
            self.barycentre_archetype = self.Omega_manifold.phi.inverse(torch.zeros_like(archetypes[0])[None])
            log_archetypes = self.Omega_manifold.log(self.barycentre_archetype[None], archetypes[None])[0,0].reshape(self.r, -1)
            self.log_archetypes = log_archetypes 

    def forward(self, x):
        """
        
        :param x: (n, d) tensor of points to project
        :return: (n, d) tensor of projected points
        """
        weights = self.w(x).T
        return self.Omega_manifold.barycentre(self.archetypes, weights=weights)

    def w(self, x):
        """
        
        :param x: (n, d) tensor of points to project
        :return: (n, r) tensor of weights for the archetypes
        """
        log_x = self.Omega_manifold.log(self.barycentre_archetype[None], x[None])[0,0].reshape(x.shape[0], -1)
        aa_solver = ArchetypalAnalysisSolver(log_x.shape[1], log_x.shape[0])
        aa_solver.V = self.log_archetypes.T
        w_x = aa_solver.predict(log_x.T).T
        return w_x