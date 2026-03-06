import torch

from src.diffeomorphisms.vector.star_gaussian import StarGaussianVectorDiffeomorphism
from src.diffeomorphisms.vector.product_starflow import ProductStarFlowVectorDiffeomorphism

class LatentStarFlowTraining(torch.nn.Module):
    def __init__(self, product_starflow_distribution, lr=1e-3):
        super(LatentStarFlowTraining, self).__init__()
        self.starflow = product_starflow_distribution
        
        self.optimizer = torch.optim.Adam(self.starflow.parameters(), lr=lr)

    def loss(self, x):
        log_prob = self.starflow.log_prob(x)  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()