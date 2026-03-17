from src.transforms.vector import VectorTransform

class TranslationVectorTransform(VectorTransform):
    def __init__(self, translation):
        super().__init__(translation.shape[0])
        self.translation = translation

    def forward(self, inputs, context=None):
        return inputs - self.translation, 0.0

    def inverse(self, inputs, context=None):
        return inputs + self.translation, 0.0