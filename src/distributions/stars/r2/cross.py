from src.distributions.stars import StarDistribution
from src.radials.multimodal.r2.cross import CrossRadial

class CrossStarDistribution(StarDistribution):
    def __init__(self, aggregation='softmax'):
        super().__init__(2, CrossRadial(aggregation=aggregation))