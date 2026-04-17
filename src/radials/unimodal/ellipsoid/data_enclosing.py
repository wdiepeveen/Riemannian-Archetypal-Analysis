from src.radials.unimodal import UniModalRadial
from src.radials.unimodal.offcentered_ellipsoid.data_enclosing import DataEnclosingOffCenteredEllipsoidRadial
from src.radials.unimodal.trimmed_ellipsoid.data_enclosing import DataEnclosingTrimmedEllipsoidRadial


class DataEnclosingEllipsoidRadial(UniModalRadial):
    def __init__(self, data, center, c=1.1, reg_param=1e-2, aggregation='min'):
        if aggregation is None:
            radial = DataEnclosingOffCenteredEllipsoidRadial(data, center, c=c, reg_param=reg_param)
        else:
            assert aggregation == 'min' or aggregation == 'softmin'
            radial = DataEnclosingTrimmedEllipsoidRadial(data, center, c=c, reg_param=reg_param, aggregation=aggregation)
        super().__init__(d=data.shape[1], radial=radial)