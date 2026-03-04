from nflows import flows

class StarFlowDistribution(flows.Flow):
    def __init__(self, transform, star_gaussian_distribution):
        super(StarFlowDistribution, self).__init__(transform, star_gaussian_distribution)
        self.d = star_gaussian_distribution.radial.d