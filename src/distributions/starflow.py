from nflows import flows

class StarFlow(flows.Flow):
    def __init__(self, transform, star_gaussian_distribution):
        super(StarFlow, self).__init__(transform, star_gaussian_distribution)