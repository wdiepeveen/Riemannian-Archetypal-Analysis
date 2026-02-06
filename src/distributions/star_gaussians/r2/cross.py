from src.distributions.star_gaussians import StarGaussianDistribution
from src.radials.multimodal.r2.cross import CrossRadial

class CrossStarGaussianDistribution(StarGaussianDistribution):
    def __init__(self):
        super().__init__(CrossRadial())