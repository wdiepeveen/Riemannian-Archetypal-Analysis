from nflows import flows

class StarFlowDistribution(flows.Flow):
    def __init__(self, d, vector_transform, star_gaussian_distribution):
        super(StarFlowDistribution, self).__init__(vector_transform, star_gaussian_distribution)
        self.d = d
        assert self.d == star_gaussian_distribution.d and self.d == vector_transform.d