import torch

from src.diffeomorphisms.vector.star_gaussian import StarGaussianVectorDiffeomorphism
from src.diffeomorphisms.vector.product import ProductVectorDiffeomorphism
from src.diffeomorphisms.identity import IdentityDiffeomorphism

class LatentStarGaussianTraining(torch.nn.Module):
    def __init__(self, diffeo, star_gaussian_distribution, lr=1e-3):
        super(LatentStarGaussianTraining, self).__init__()
        self.phi = diffeo
        self.star = star_gaussian_distribution
        
        self.optimizer = torch.optim.Adam(self.star.parameters(), lr=lr)

    @property
    def psi_o_phi(self):
        return self.starflow_diffeo

    @property
    def psi(self):
        if self.phi.d == self.star.d:
            return StarGaussianVectorDiffeomorphism(self.phi.d, self.star)
        else:
            return ProductVectorDiffeomorphism([StarGaussianVectorDiffeomorphism(self.star.d, self.star), IdentityDiffeomorphism(self.phi.d - self.star.d)])    

    def loss(self, x):
        phi_x = self.phi.forward(x).reshape(x.shape[0], -1)  # N x d
        log_prob = self.star.log_prob(phi_x[:,:self.star.d])  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()