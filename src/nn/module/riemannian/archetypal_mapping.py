import torch

class RiemannianArchetypalMapping(torch.nn.Module):
    def __init__(self, Phi_manifold, archetypes, phi_manifold=None, beta=1.):
        super().__init__()
        self.Phi_manifold = Phi_manifold
        self.archetypes = archetypes
        if phi_manifold is None:
            self.phi_manifold = 5# standard euclidean manifold on the same dimension as the archetypes
        else:
            self.phi_manifold = phi_manifold
        self.beta = beta # inverse temperature for the softmax in the weights computation

        self.r = archetypes.shape[0]

        with torch.no_grad():
            self.barycentre_archetype = self.Phi_manifold.barycentre(archetypes)
            log_archetypes = self.phi_manifold.log(self.barycentre_archetype[None], archetypes[None])[0,0].reshape(self.r, -1)
            self.log_archetypes = log_archetypes / log_archetypes.norm(dim=1, keepdim=True)

    def forward(self, x):
        w_x = self.w(x)
        return self.Phi_manifold.barycentre(self.archetypes, weights=w_x)


    def w(self, x):
        log_x = self.phi_manifold.log(self.barycentre_archetype[None], x[None])[0,0].reshape(x.shape[0], -1)
        w_x = torch.einsum('Ni,ri->Nr', log_x, self.log_archetypes).T
        w_x = torch.nn.functional.softmax(self.beta * w_x, dim=0)
        return w_x