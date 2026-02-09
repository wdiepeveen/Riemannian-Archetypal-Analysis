from src.radials import Radial

class UniModalRadial(Radial):
    def __init__(self, d):
        super().__init__(d)

    def forward(self, theta):
        raise NotImplementedError("UniModalRadial is an abstract base class; implement forward in a subclass")