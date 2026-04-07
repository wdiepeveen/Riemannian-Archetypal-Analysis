from src.distributions.stars import StarDistribution
from src.radials.multimodal.ellipsoid.centered import MultiCenteredEllipsoidRadial
from src.radials.multimodal.ellipsoid.offcentered import MultiOffCenteredEllipsoidRadial
from src.radials.multimodal.ellipsoid.offcentered_trimmed import MultiOffCenteredTrimmedEllipsoidRadial

class MultiModalEllipsoidStarDistribution(StarDistribution):
    def __init__(self, covs, mus=None, c=4/3, p=None, aggregation='max', trimmed=False, trim_aggregation='min'):
        if mus is None:
            radial = MultiCenteredEllipsoidRadial(covs, p=p, aggregation=aggregation)
        else:
            if trimmed:
                radial = MultiOffCenteredTrimmedEllipsoidRadial(covs, mus=mus, c=c, p=p, aggregation=aggregation, soft_trim_aggregation=(trim_aggregation=='softmin'))
            else:
                radial = MultiOffCenteredEllipsoidRadial(covs, mus=mus, c=c, p=p, aggregation=aggregation)
        d = covs[0].shape[0]
        super().__init__(d, radial)