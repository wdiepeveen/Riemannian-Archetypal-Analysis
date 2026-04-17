from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.offcentered_ellipsoid.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.data_enclosing import DataEnclosingTrimmedEllipsoidRadial


class MultiDataEnclosingEllipsoidRadial(MultiModalRadial):
    def __init__(self, datas, centers, c=1.1, reg_param=1e-2, outer_aggregation='max', inner_aggregation='min'):
        if inner_aggregation is None:
            radials = [DataEnclosingOffCenteredEllipsoidRadial(data, center, c=c, reg_param=reg_param) for data, center in zip(datas, centers)]
        else:
            assert inner_aggregation == 'min' or inner_aggregation == 'softmin'
            radials = [DataEnclosingTrimmedEllipsoidRadial(data, center, c=c, reg_param=reg_param, aggregation=inner_aggregation) for data, center in zip(datas, centers)]
        super().__init__(d=datas[0].shape[1], radials=radials, aggregation=outer_aggregation)