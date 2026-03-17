from src.diffeomorphisms.image import ImageDiffeomorphism
from src.diffeomorphisms.image.to_vec import ToVecImageDiffeomorphism
from src.diffeomorphisms.vector.star import StarVectorDiffeomorphism

class StarImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width, star_distribution, s=1.):
        super().__init__(in_channels, height, width)
        assert star_distribution.d == in_channels * height * width, "Distribution dimension must match diffeomorphism dimension."

        self.star_vector_diffeo = StarVectorDiffeomorphism(self.d, star_distribution, s=s)
        self.to_vec = ToVecImageDiffeomorphism(in_channels, height, width)
    
    def forward(self, x):
        return self.to_vec.inverse(self.star_vector_diffeo.forward(self.to_vec(x)))

    def inverse(self, y):
        return self.to_vec.inverse(self.star_vector_diffeo.inverse(self.to_vec(y)))

    def differential_forward(self, x, X):
        return self.to_vec.inverse(self.star_vector_diffeo.differential_forward(self.to_vec(x), self.to_vec(X)))

    def differential_inverse(self, y, Y):
        return self.to_vec.inverse(self.star_vector_diffeo.differential_inverse(self.to_vec(y), self.to_vec(Y)))