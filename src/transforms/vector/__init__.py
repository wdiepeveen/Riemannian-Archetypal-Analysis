from nflows.transforms import Transform

class VectorTransform(Transform):
    def __init__(self, d):
        super().__init__()
        self.d = d