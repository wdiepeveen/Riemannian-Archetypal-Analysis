import torch

class DenoisingScoreMatchingTraining(torch.nn.Module):
    def __init__(self, starflow_diffeo, lr=1e-3):
        super(DenoisingScoreMatchingTraining, self).__init__()
        self.starflow_diffeo = starflow_diffeo # star flow with constant det Jac for the diffeomorphism phi
        
        self.optimizer = torch.optim.Adam(self.starflow_diffeo.parameters(), lr=lr)

    @property
    def psi(self):
        return 6
    
    @property
    def psi(self):
        return 5
    
    def score(self, x):
        return - self.starflow_diffeo.adjoint_differential_forward(x, self.starflow_diffeo.forward(x))
        
    def loss(self, x, sigma=0.1):
        N = x.shape[0]
        eps = torch.randn_like(x)
        x_tilde = x + sigma * eps
        score_x_tilde = self.score(x_tilde)  
        return ((sigma * score_x_tilde + eps) ** 2).reshape(N, -1).sum(dim=-1).mean()
    
    def train_step(self, x, sigma=0.1):
        self.optimizer.zero_grad()
        loss = self.loss(x, sigma)
        loss.backward()
        self.optimizer.step()
        return loss.item()