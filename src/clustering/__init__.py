class Clustering:
    """Base clustering class."""
    def __init__(self, euclidean):
        self.euclidean = euclidean

    def fit(self, x):
        raise NotImplementedError(
            "Subclasses should implement this"
            )

    def predict(self, x):
        raise NotImplementedError(
            "Subclasses should implement this"
        )