from src.distributions.starflows import StarFlowDistribution
from src.transforms.vector.translation import TranslationVectorTransform

class RecenteredStarFlowDistribution(StarFlowDistribution):
    def __init__(self, center, star_gaussian_distribution):
        super().__init__(star_gaussian_distribution.d, TranslationVectorTransform(center), star_gaussian_distribution)
        self.center = center