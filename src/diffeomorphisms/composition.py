from src.diffeomorphisms import Diffeomorphism

class CompositionDiffeomorphism(Diffeomorphism):
    def __init__(self, diffeos):
        """
        Composition of multiple diffeomorphisms
        :param diffeos: List[Diffeomorphism]
        """
        assert all(diffeo.d == diffeos[0].d for diffeo in diffeos), "Diffeomorphisms must have the same dimension"
        super().__init__(diffeos[0].d)
        self.diffeos = diffeos

    def forward(self, x):
        out = x
        for diffeo in self.diffeos:
            out = diffeo.forward(out)
        return out  
    
    def inverse(self, y):
        out = y
        for diffeo in reversed(self.diffeos):
            out = diffeo.inverse(out)
        return out
    
    def differential_forward(self, x, X):
        out = X
        current_x = x
        for diffeo in self.diffeos:
            out = diffeo.differential_forward(current_x, out)
            current_x = diffeo.forward(current_x)
        return out
    
    def differential_inverse(self, y, Y):
        out = Y
        current_y = y
        for diffeo in reversed(self.diffeos):
            out = diffeo.differential_inverse(current_y, out)
            current_y = diffeo.inverse(current_y)
        return out
