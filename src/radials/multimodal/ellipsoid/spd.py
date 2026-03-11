from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.elliposoid.spd import SPDEllipsoidUniModalRadial

class MultiSPDEllipsoidRadial(MultiModalRadial):
    def __init__(self, d, Sigmas, signs, aggregation='sum'):
        super().__init__(d, [SPDEllipsoidUniModalRadial(Sigma, sign=sign) for Sigma, sign in zip(Sigmas, signs)], aggregation=aggregation)