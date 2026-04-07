
from src.distributions.starflows import StarFlowDistribution
from src.distributions.stars.r2.cross import CrossStarDistribution

from src.transforms.vector.r2.river import RiverVectorTransform

class RiverCrossStarFlowDistribution(StarFlowDistribution):
    def __init__(self):
        super(RiverCrossStarFlowDistribution, self).__init__(2, RiverVectorTransform(2.,0), CrossStarDistribution())