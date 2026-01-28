from src.diffeomorphisms.vector import VectorDiffeomorphism

class ProductEmbeddingVectorDiffeomorphism(VectorDiffeomorphism):
    """ Base class describing a Diffeomorphism that embeds a vector into a product vector space """

    def __init__(self, d_list):
        super().__init__(sum(d_list))
        self.d_list = d_list
        self.m = len(d_list)