import torch

from src.diffeomorphisms.vector.star_gaussian import StarGaussianVectorDiffeomorphism

class StarGaussianTraining(torch.nn.Module):
    def __init__(self, diffeo, star_gaussian, lr=1e-3):
        super(StarGaussianTraining, self).__init__()
        assert diffeo.d == star_gaussian.d, "Diffeomorphism and Star Gaussian must have the same dimension"
        self.phi = diffeo
        self.star = star_gaussian
        
        self.optimizer = torch.optim.Adam(self.star.parameters(), lr=lr)

    @property
    def psi(self):
        return StarGaussianVectorDiffeomorphism(self.phi.d, self.star)

    def loss(self, x):
        phi_x = self.phi.forward(x).reshape(x.shape[0], -1)  # N x d
        log_prob = self.star.log_prob(phi_x)  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()