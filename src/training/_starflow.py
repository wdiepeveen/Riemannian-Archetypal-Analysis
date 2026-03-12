import torch

class StarFlowTraining(torch.nn.Module):
    def __init__(self, starflow_diffeo, r=None, reg_param=1e-4, lr=1e-3):
        super(StarFlowTraining, self).__init__()
        self.starflow_diffeo = starflow_diffeo

        self.r = r
        self.reg_param = reg_param
        
        self.optimizer = torch.optim.Adam(self.starflow_diffeo.parameters(), lr=lr)

    @property
    def psi_o_phi(self):
        return self.starflow_diffeo
    
    @property
    def phi(self):
        return self.starflow_diffeo.transform

    def loss(self, x):
        # compute negative log-likelihood of the data under the star flow model
        phi_x = self.phi.forward(x).reshape(x.shape[0], -1)  # N x d
        log_prob = self.starflow_diffeo.starflow.log_prob(phi_x)  # N
        loss = -log_prob.mean()

        # compute low rank regularization term
        if self.r is not None:
            mask = torch.zeros_like(phi_x)
            mask[:, :self.r] = 1.0
            x_rec = self.phi.inverse(mask * phi_x)
            reg_loss = self.reg_param * ((x - x_rec) ** 2).sum(dim=-1).mean()
            
            loss += reg_loss
        return loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()