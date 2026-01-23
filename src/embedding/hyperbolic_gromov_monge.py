import torch

from src.diffeomorphisms.vector.poincare import PoincareVectorDiffeomorphism
from src.embedding import Embedding
from src.manifolds.hyperbolic.vector.standard import StandardVectorHyperbolic

class HyperbolicGromovMongeEmbedding(Embedding):
    def __init__(self, nn_model, output_dim=2):
        super(HyperbolicGromovMongeEmbedding, self).__init__(output_dim)
        self.nn_model = nn_model
        self.poincare_map = PoincareVectorDiffeomorphism(output_dim)
        self.manifold = StandardVectorHyperbolic(output_dim)
        self.origin = None

    def forward(self, x):
        """
        
        :param x: N x [data_dims]
        :return: N x output_dim
        """
        nn_output = self.nn_model(x)
        hyperbolic_embedding = self.poincare_map(nn_output)
        if self.origin is not None:
            hyperbolic_embedding = self.translate(hyperbolic_embedding, self.origin)
        return hyperbolic_embedding
    
    def set_origin(self, origin):
        """
        
        :param origin: output_dim 
        """
        self.origin = origin

    def translate(self, z, translation):
        """
        
        :param z: N x output_dim
        :param translation: output_dim
        :return: N x output_dim
        """
        norm_tr_sq = torch.sum(translation**2)
        norm_x_sq = torch.sum(z**2, dim=-1)
        xtr_dot = torch.sum(z * translation.unsqueeze(0), dim=-1)
        numerator = (1 - norm_tr_sq) * z - (1 - 2 * xtr_dot.unsqueeze(-1) + norm_x_sq.unsqueeze(-1)) * translation.unsqueeze(0)
        denominator = 1 - 2 * xtr_dot + norm_tr_sq * norm_x_sq
        return numerator / denominator.unsqueeze(-1)

    def distance(self, x1, x2):
        """
        
        :param x1: N x [data_dims]
        :param x2: N x [data_dims]
        :return: N
        """
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        dist = self.manifold.distance(emb1.unsqueeze(1), emb2.unsqueeze(1)).squeeze(2).squeeze(1)
        return dist
    
    def interpolate(self, x1, x2, t):
        """
        
        :param x1: N x [data_dims]
        :param x2: N x [data_dims]
        :param t: K
        :return: N x K x output_dim
        """
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        interp_emb = self.manifold.geodesic(emb1.unsqueeze(1), emb2.unsqueeze(1), t).squeeze(2).squeeze(1)
        return interp_emb
    
    def barycentre(self, x):
        """
        
        :param x: N x [data_dims]
        :return: output_dim
        """
        emb_points = self.forward(x)
        barycentre = self.manifold.barycentre(emb_points, max_iter=500, tol=1e-5, step_size=0.5)
        return barycentre