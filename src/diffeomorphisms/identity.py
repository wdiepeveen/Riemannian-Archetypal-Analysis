from src.diffeomorphisms import Diffeomorphism

class IdentityDiffeomorphism(Diffeomorphism):
    def __init__(self, d):
        super().__init__(d)

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def differential_forward(self, x, X):
        return X
    
    def differential_inverse(self, y, Y):
        return Y