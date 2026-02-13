import torch

class GromovMongeEmbeddingInformedStarFlowTraining(torch.nn.Module):
    def __init__(self, embedding_model, starflow_diffeomorphism, reg_param=1e-4, sigma=None):
        super(GromovMongeEmbeddingInformedStarFlowTraining, self).__init__()
        assert sigma is not None, "Sigma must be provided for latent variance regularization"
        self.emb = embedding_model # embedding model into lower-dimensional Euclidean space
        self.star_flow_diffeo = starflow_diffeomorphism # star flow model
        self.reg_param = reg_param
        self.sigma = sigma

    # def loss(self, x):
    #     """
        
    #     :param x: N x [data_dims]
    #     :return: scalar loss
    #     """
    #     return self.recon_loss(x) + (self.reg_param * self.neg_log_prob_loss(x) if self.reg_param is not None else 0.0)
        
        # # reconstruction loss
        # emb_x = self.emb(x)  # N x emb.d
        # log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
        # p_x = torch.zeros((x.shape[0], self.star_flow_diffeo.d), device=x.device, dtype=x.dtype)  # N x d
        # p_x[:, :self.emb.d] = log_0_emb_x
        # phi_inv_p_x = self.star_flow_diffeo.transform.inverse(p_x.reshape(x.shape))  # N x d
        # recon_loss = torch.nn.functional.mse_loss(phi_inv_p_x, x, reduction='mean')

        # # negative log likelihood loss
        # if self.reg_param is None:
        #     return recon_loss
        # log_prob = self.star_flow_diffeo.star_flow.log_prob(x)  # N
        # neg_log_prob_loss = -log_prob.mean()

        # # total loss
        # loss = recon_loss + self.reg_param * neg_log_prob_loss
        # return loss
    
    def transform_loss(self, x):
        # reconstruction loss
        emb_x = self.emb(x)  # N x emb.d
        log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
        p_x = torch.zeros((x.shape[0], self.star_flow_diffeo.d), device=x.device, dtype=x.dtype)  # N x d
        p_x[:, :self.emb.d] = log_0_emb_x
        phi_inv_p_x = self.star_flow_diffeo.transform.inverse(p_x.reshape(x.shape))  # N x d
        recon_loss = torch.nn.functional.mse_loss(phi_inv_p_x, x, reduction='mean')

        # latent variance loss
        phi_x = self.star_flow_diffeo.transform(x).reshape(x.shape[0], -1)  # N x d
        phi_x_noise = phi_x[:, self.emb.d:] # N x (d - emb.d)
        latent_loss = torch.sum(torch.log(torch.mean(phi_x_noise**2, dim=0) / (self.sigma**2))**2)

        return recon_loss + self.reg_param * latent_loss
    
    def radial_loss(self, x):
        log_prob = self.star_flow_diffeo.star_flow.log_prob(x)  # N
        neg_log_prob_loss = -log_prob.mean()
        return neg_log_prob_loss
    
    # def latent_var_loss(self, x, sigma=0.1):
    #     phi_x = self.star_flow_diffeo.transform(x).reshape(x.shape[0], -1)  # N x d
    #     phi_x_noise = phi_x[:, self.emb.d:] # N x (d - emb.d)
    #     latent_loss = torch.sum(torch.log(torch.mean(phi_x_noise**2, dim=0) / (sigma**2))**2)
    #     return latent_loss
