import torch

class HyperbolicGromovMongeEmbeddingInformedNFlowTraining(torch.nn.Module):
    def __init__(self, embedding_model, nflow_diffeomorphism, reg_param=1e-3):
        super(HyperbolicGromovMongeEmbeddingInformedNFlowTraining, self).__init__()
        self.emb = embedding_model # embedding model into hyperbolic space
        self.phi = nflow_diffeomorphism # normalizing flow model into Euclidean space
        self.reg_param = reg_param
        self.W = torch.nn.Parameter(torch.randn(self.emb.d, self.phi.d))

    def loss(self, x):
        """
        
        :param x: N x [data_dims]
        :return: scalar loss
        """
        
        # reconstruction loss
        emb_x = self.emb(x)  # N x emb.d
        log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
        # pad log_0_emb_x with zeros to get a vector in R^d
        p_x = torch.zeros_like(x)  # N x d
        p_x[:, :self.emb.d] = log_0_emb_x
        # p_x = log_0_emb_x @ self.W  # N x d
        phi_inv_p_x = self.phi.inverse(p_x.reshape(x.shape))  # N x d
        recon_loss = torch.nn.functional.mse_loss(phi_inv_p_x, x, reduction='mean')

        # negative log likelihood loss
        log_prob = self.phi.nflow.log_prob(x)  # N
        neg_log_prob_loss = -log_prob.mean()
        
        # total loss
        loss = recon_loss + self.reg_param * neg_log_prob_loss
        return loss