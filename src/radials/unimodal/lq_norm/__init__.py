from src.radials.unimodal import UniModalRadial

class LqNormRadial(UniModalRadial):
    def __init__(self, d, q: float, alpha: float = 1.0):
        super().__init__(d)
        self.q = q
        self.alpha = alpha

    def forward(self, theta: float) -> float:
        return self.alpha * (abs(theta) ** self.q).sum(-1) ** (-1 / self.q)