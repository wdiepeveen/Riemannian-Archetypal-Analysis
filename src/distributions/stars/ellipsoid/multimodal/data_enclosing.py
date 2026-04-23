from src.distributions.stars import StarDistribution
from src.radials.multimodal.ellipsoid.data_enclosing import MultiDataEnclosingEllipsoidRadial

class MultimodalDataEnclosingEllipsoidStarDistribution(StarDistribution):
    def __init__(self, datas, centers, c=1.1, reg_param=1e-2, outer_aggregation='softmax', inner_aggregation='softmin'):
        d = datas[0].shape[1]
        radial = MultiDataEnclosingEllipsoidRadial(datas, centers, c=c, reg_param=reg_param, outer_aggregation=outer_aggregation, inner_aggregation=inner_aggregation)
        super().__init__(d, radial)
