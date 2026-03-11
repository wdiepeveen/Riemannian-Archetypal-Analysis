from src.radials.unimodal import UniModalRadial

class EllipsoidUniModalRadial(UniModalRadial):
    """ 
    Implementation of a unimodal radial with an ellipsoid shape. The radial is defined as: 
        rho(theta) = a * (1 - eps^2) / (1 - eps * q \cdot L(theta)/ \|L(theta)\|), 
    where a is the semi-major axis, eps is the eccentricity, and q is the axis direction. 
    The linear transformation L(theta) is defined in the subclasses.
    """
    def __init__(self, d, semi_major_axis, eccentricity, axis_direction):
        super().__init__(d)
        self.a = semi_major_axis
        self.eps = eccentricity
        self.q = axis_direction

    def forward(self, theta):
        unit_q = self.q / self.q.norm(2)
        linear_theta = self.linear(theta)
        unit_linear_theta = linear_theta / linear_theta.norm(2, dim=-1, keepdim=True)
        cos = (unit_q[None] * unit_linear_theta).sum(dim=-1).clamp(-1, 1)
        return self.a * (1 - self.eps ** 2) / (1 - self.eps * cos)

    def linear(self, theta):
        raise NotImplementedError("Subclasses should implement this method to return the linear transformation of theta.")
