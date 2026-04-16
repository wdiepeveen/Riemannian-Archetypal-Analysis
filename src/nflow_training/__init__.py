import torch

class NFlowTraining:
    def __init__(self, nflow_distribution, lr=1e-3):
        self.nflow = nflow_distribution
        
        self.optimizer = torch.optim.Adam(self.nflow.parameters(), lr=lr)

    def loss(self, x):
        log_prob = self.nflow.log_prob(x)  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()