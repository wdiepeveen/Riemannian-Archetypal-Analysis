import torch

class GromovMongeEmbeddingInformedStarFlowTraining(torch.nn.Module):
    def __init__(self, embedding_model, starflow_diffeomorphism, reg_param=None):
        super(GromovMongeEmbeddingInformedStarFlowTraining, self).__init__()
        self.emb = embedding_model # embedding model into lower-dimensional Euclidean space
        self.star_flow_diffeo = starflow_diffeomorphism # star flow model
        self.reg_param = reg_param

    def loss(self, x):
        """
        
        :param x: N x [data_dims]
        :return: scalar loss
        """
        
        # reconstruction loss
        emb_x = self.emb(x)  # N x emb.d
        log_0_emb_x = self.emb.poincare_map.inverse(emb_x)  # N x emb.d
        p_x = torch.zeros_like(x)  # N x d
        p_x[:, :self.emb.d] = log_0_emb_x
        phi_inv_p_x = self.star_flow_diffeo.transform.inverse(p_x.reshape(x.shape))  # N x d
        recon_loss = torch.nn.functional.mse_loss(phi_inv_p_x, x, reduction='mean')

        # negative log likelihood loss
        if self.reg_param is None:
            return recon_loss
        log_prob = self.star_flow_diffeo.star_flow.log_prob(x)  # N
        neg_log_prob_loss = -log_prob.mean()

        # total loss
        loss = recon_loss + self.reg_param * neg_log_prob_loss
        return loss