from src.distributions.stars import StarDistribution
from src.radials.unimodal.ellipsoid.data_enclosing import DataEnclosingEllipsoidRadial

class UnimodalDataEnclosingEllipsoidStarDistribution(StarDistribution):
    def __init__(self, data, center, c=1.1, reg_param=1e-2, aggregation='min'):
        d = data.shape[1]
        radial = DataEnclosingEllipsoidRadial(data, center, c=c, reg_param=reg_param, aggregation=aggregation)
        super().__init__(d, radial)
