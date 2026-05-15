from src.distributions.stars import StarDistribution
from src.radials.multimodal.ellipsoid.data_enclosing import MultiDataEnclosingEllipsoidRadial

class MultimodalDataEnclosingEllipsoidStarDistribution(StarDistribution):
    def __init__(self, datas, centers, alpha=1.1, beta=0.1, outer_aggregation='softmax', inner_aggregation='softmin'):
        d = datas[0].shape[1]
        radial = MultiDataEnclosingEllipsoidRadial(datas, centers, alpha=alpha, beta=beta, outer_aggregation=outer_aggregation, inner_aggregation=inner_aggregation)
        super().__init__(d, radial)
