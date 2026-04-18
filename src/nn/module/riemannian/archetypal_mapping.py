import torch

from archetypes import AA

# from src.dimension_reduction.archetypal_analysis import ArchetypalAnalysisSolver

class RiemannianArchetypalMapping(torch.nn.Module):
    def __init__(self, Omega_manifold, archetypes):
        super().__init__()
        self.Omega_manifold = Omega_manifold
        
        self.archetypes = archetypes
        self.r = archetypes.shape[0]

        with torch.no_grad():
            self.barycentre_archetype = self.Omega_manifold.phi.inverse(torch.zeros_like(archetypes[0])[None])
            self.log_archetypes = self.Omega_manifold.log(self.barycentre_archetype[None], archetypes[None])[0,0].reshape(self.r, -1)

        self.aa = AA(self.r, init='furthest_sum')
        self.aa.fit(self.log_archetypes.detach())
        self.perm = self.find_perm(self.log_archetypes, torch.from_numpy(self.aa.archetypes_).to(self.log_archetypes.dtype))

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
        w_x = torch.from_numpy(self.aa.transform(log_x)).to(x.dtype)[:, self.perm]
        return w_x
    
    def find_perm(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Find permutation p such that X[i] ~= Y[p[i]] using cosine similarity.
        Assumes Y is a unique permutation of (noisy) rows of X.
        """
        # Normalize rows to unit norm
        Xn = torch.nn.functional.normalize(X, p=2, dim=1)
        Yn = torch.nn.functional.normalize(Y, p=2, dim=1)

        # Similarity matrix: (n, n), sim[i, j] = <X_i, Y_j> / (||X_i|| ||Y_j||)
        sim = Xn @ Yn.T   # normalized inner products

        n = X.size(0)
        p = torch.empty(n, dtype=torch.long, device=X.device)

        # Greedy 1-1 assignment
        assigned_Y = torch.zeros(n, dtype=torch.bool, device=X.device)
        assigned_X = torch.zeros(n, dtype=torch.bool, device=X.device)

        for _ in range(n):
            # On each step, find the largest remaining similarity
            sim_masked = sim.clone()
            sim_masked[assigned_X] = -float("inf")
            sim_masked[:, assigned_Y] = -float("inf")

            i, j = torch.nonzero(sim_masked == sim_masked.max(), as_tuple=True)
            i = i[0]
            j = j[0]

            p[i] = j
            assigned_X[i] = True
            assigned_Y[j] = True

        return p