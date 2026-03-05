from src.radials.multimodal.inverse_affine_norm import MultiInverseAffineNormRadial
from src.radials.unimodal.inverse_affine_norm.linear import InverseLinearNormRadial

class MultiInverseLinearNormRadial(MultiInverseAffineNormRadial):
    def __init__(self, d, num_radials, r=None, sigma=1e-3):
        if r is not None:
            # r is an integer or r is a list
            if isinstance(r, int):
                assert d >= r > 0, f"r must be a positive integer less than or equal to d, but got r={r} and d={d}"
                r = [r] * num_radials
            elif isinstance(r, list):
                if len(r) != num_radials:
                    raise ValueError(f"Length of r must be equal to num_radials, but got len(r)={len(r)} and num_radials={num_radials}")
            else:
                raise ValueError(f"r must be an integer or a list, but got r={r} of type {type(r)}")
        else:
            r = [None] * num_radials

        super().__init__(d, [InverseLinearNormRadial(d, r=r[i], sigma=sigma) for i in range(num_radials)])
    
    def mode_variance(self, idx):
        """
        Return the variance of the idx-th mode of the multimodal radial.
        :param idx: int
        :return: tensor of shape (d, d)
        """
        return self.radials[idx].mode_variance