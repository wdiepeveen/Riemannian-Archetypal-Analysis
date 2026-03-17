from nflows.transforms import CompositeTransform

from src.distributions.starflows import StarFlowDistribution
from src.transforms.vector.translation import TranslationVectorTransform
from src.transforms.image.translation import TranslationImageTransform
from src.transforms.image.to_vec import ToVecImageTransform

class RecenteredStarFlowDistribution(StarFlowDistribution):
    def __init__(self, center, star_gaussian_distribution):
        # check whether center is vector or image data
        if center.ndim == 1:
            translation_transform = TranslationVectorTransform(center)
        if center.ndim == 3:
            translation_transform = CompositeTransform([TranslationImageTransform(center), ToVecImageTransform(center.shape[0], center.shape[1], center.shape[2])])
        super().__init__(star_gaussian_distribution.d, translation_transform, star_gaussian_distribution)
        self.center = center