from src.diffeomorphisms.composition import CompositionDiffeomorphism

class ImageCompositionDiffeomorphism(CompositionDiffeomorphism):
    """ Composition of diffeomorphisms acting on images. """

    def __init__(self, diffeomorphisms, in_channels, height, width):
        super().__init__(diffeomorphisms)

        self.C = in_channels
        self.H = height
        self.W = width