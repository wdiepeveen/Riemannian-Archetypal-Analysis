from nflows.transforms import CompositeTransform

from src.distributions.starflows import StarFlowDistribution
from src.transforms.vector.translation import TranslationVectorTransform
from src.transforms.vector.linear.diagonal import DiagonalLinearVectorTransform
from src.transforms.image.translation import TranslationImageTransform
from src.transforms.image.linear.diagonal import DiagonalLinearImageTransform
from src.transforms.image.to_vec import ToVecImageTransform

class NormalizedStarFlowDistribution(StarFlowDistribution):
    def __init__(self, star_gaussian_distribution, mean, std):
        # check whether mean is vector or image data
        if mean.ndim == 1:
            translation_transform = TranslationVectorTransform(mean)
            rescaling_transform = DiagonalLinearVectorTransform(1 / std)  # to ensure that the inverse also normalizes
            transform = CompositeTransform([translation_transform, rescaling_transform])
        elif mean.ndim == 3:
            translation_transform = TranslationImageTransform(mean)
            rescaling_transform = DiagonalLinearImageTransform(1 / std)  # to ensure that the inverse also normalizes
            transform = CompositeTransform([translation_transform, rescaling_transform, ToVecImageTransform(mean.shape[0], mean.shape[1], mean.shape[2])])
        else:
            raise ValueError(f"Mean has unsupported number of dimensions: {mean.ndim}")
        super().__init__(star_gaussian_distribution.d, transform, star_gaussian_distribution)
        self.center = mean
        self.std = std