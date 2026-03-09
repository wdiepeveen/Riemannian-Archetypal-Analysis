import torch

from src.diffeomorphisms.vector.star_gaussian import StarGaussianVectorDiffeomorphism
from src.diffeomorphisms.vector.product import ProductVectorDiffeomorphism
from src.diffeomorphisms.identity import IdentityDiffeomorphism

class LatentStarGaussianTraining(torch.nn.Module):
    def __init__(self, embedding, star_gaussian_distribution, lr=1e-3):
        super(LatentStarGaussianTraining, self).__init__()
        self.emb = embedding
        self.star = star_gaussian_distribution
        
        self.optimizer = torch.optim.Adam(self.star.parameters(), lr=lr)

    def psi(self, d):
        if d == self.star.d:
            return StarGaussianVectorDiffeomorphism(d, self.star)
        else:
            return ProductVectorDiffeomorphism([StarGaussianVectorDiffeomorphism(self.star.d, self.star), IdentityDiffeomorphism(d - self.star.d)])    

    def loss(self, x):
        phi_x = self.emb.forward(x).reshape(x.shape[0], -1)  # N x d
        log_prob = self.star.log_prob(phi_x[:,:self.star.d])  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()