from nflows import flows

class StarFlowDistribution(flows.Flow):
    def __init__(self, d, transform, star_gaussian_distribution):
        super(StarFlowDistribution, self).__init__(transform, star_gaussian_distribution)
        self.d = d
        assert self.d == star_gaussian_distribution.d