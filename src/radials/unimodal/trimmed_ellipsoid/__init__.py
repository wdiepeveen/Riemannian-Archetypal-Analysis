from src.radials.unimodal import UniModalRadial
from src.radials.multimodal.intersected import IntersectedRadial

class TrimmedEllipsoidRadial(UniModalRadial):
    def __init__(self, d, ellipsoid_radial, trim_radial, aggregation='softmin'):
        super().__init__(d)
        self.ellipsoid_radial = ellipsoid_radial
        self.trim_radial = trim_radial
        self.radial = IntersectedRadial([ellipsoid_radial, trim_radial], aggregation=aggregation)
        
    def forward(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        return self.radial(theta)

