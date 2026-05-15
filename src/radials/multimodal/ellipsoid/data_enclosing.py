from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.offcentered_ellipsoid.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.data_enclosing import DataEnclosingTrimmedEllipsoidRadial


class MultiDataEnclosingEllipsoidRadial(MultiModalRadial):
    def __init__(self, datas, centers, alpha=1.1, beta=0.1, outer_aggregation='softmax', inner_aggregation='softmin'):
        if inner_aggregation is None:
            radials = [DataEnclosingOffCenteredEllipsoidRadial(data, center, alpha=alpha, beta=beta) for data, center in zip(datas, centers)]
        else:
            assert inner_aggregation == 'min' or inner_aggregation == 'softmin'
            radials = [DataEnclosingTrimmedEllipsoidRadial(data, center, alpha=alpha, beta=beta, aggregation=inner_aggregation) for data, center in zip(datas, centers)]
        super().__init__(d=datas[0].shape[1], radials=radials, aggregation=outer_aggregation)