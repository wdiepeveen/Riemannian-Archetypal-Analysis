import torch

class DiffeomorphicDecoderTraining(torch.nn.Module):
    def __init__(self, embedding, diffeomorphic_decoder, reg_param=1e-4, sigma=None, lr=1e-3):
        super(DiffeomorphicDecoderTraining, self).__init__()
        assert sigma is not None, "Sigma must be provided for latent variance regularization"

        self.emb = embedding
        self.phi = diffeomorphic_decoder
        self.reg_param = reg_param
        self.sigma = sigma

        self.optimizer = torch.optim.Adam(self.phi.parameters(), lr=lr)

    def loss(self, x):
        # reconstruction loss
        emb_x = self.emb(x)  # N x emb.d
        log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
        p_x = torch.zeros((x.shape[0], self.phi.d), device=x.device, dtype=x.dtype)  # N x d
        p_x[:, :self.emb.d] = log_0_emb_x
        phi_inv_p_x = self.phi.inverse(p_x.reshape(x.shape))  # N x d
        recon_loss = torch.nn.functional.mse_loss(phi_inv_p_x, x, reduction='mean')
    
        # latent variance loss
        phi_x = self.phi.forward(x).reshape(x.shape[0], -1)  # N x d
        phi_x_noise = phi_x[:, self.emb.d:] # N x (d - emb.d)
        latent_loss = torch.sum(torch.log(torch.mean(phi_x_noise**2, dim=0) / (self.sigma**2))**2)

        return recon_loss + self.reg_param * latent_loss
    
    def train_step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reconstruct(self, x):
        with torch.no_grad():
            emb_x = self.emb(x)  # N x emb.d
            log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
            p_x = torch.zeros((x.shape[0], self.phi.d), device=x.device, dtype=x.dtype)  # N x d
            p_x[:, :self.emb.d] = log_0_emb_x
            phi_inv_p_x = self.phi.inverse(p_x.reshape(x.shape))  # N x d
            return phi_inv_p_x