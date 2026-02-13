import torch

class HyperbolicGromovMongeEmbeddingTraining(torch.nn.Module):
    def __init__(self, hyperbolic_gromov_monge_embedding, reg_param=1e-2, lr=1e-3):
        super(HyperbolicGromovMongeEmbeddingTraining, self).__init__()
        self.emb = hyperbolic_gromov_monge_embedding
        self.reg_param = reg_param

        self.optimizer = torch.optim.Adam(self.emb.nn_model.parameters(), lr=lr)

    def loss(self, x, x_prime):
        # compute distances in data space and embedding space
        dist_data = torch.norm(x.reshape(x.shape[0], -1) - x_prime.reshape(x_prime.shape[0], -1), dim=1)
        dist_emb = self.emb.distance(x, x_prime)
        emb_loss = torch.log((1 + dist_emb ** 2) / (1 + dist_data ** 2)).pow(2).mean()

        # compute regularizer
        nn_x_mean = self.emb.nn_model(x).mean(dim=0)
        nn_x_prime_mean = self.emb.nn_model(x_prime).mean(dim=0)
        reg_loss = nn_x_mean.pow(2).sum() + nn_x_prime_mean.pow(2).sum()

        return emb_loss + self.reg_param * reg_loss
    
    def train_step(self, x, x_prime):
        self.optimizer.zero_grad()
        loss = self.loss(x, x_prime)
        loss.backward()
        self.optimizer.step()
        return loss.item()