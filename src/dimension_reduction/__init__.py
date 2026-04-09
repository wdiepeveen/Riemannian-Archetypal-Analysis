class DimensionReductionSolver:
    """ Base class for dimension reduction of d-dimensional data X = (X_1, ..., X_n) \in R^{d x n} """
    def __init__(self, d, N) -> None:
        self.d = d
        self.N = N

    def fit(self, X):
        raise NotImplementedError(
            "Subclasses should implement this"
        )