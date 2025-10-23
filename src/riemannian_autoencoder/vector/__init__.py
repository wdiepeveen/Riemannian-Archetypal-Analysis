from src.riemannian_autoencoder import RiemannianAutoencoder

class VectorRiemannianAutoencoder(RiemannianAutoencoder): # TODO move standard encoding and decoding here
    def __init__(self, vector_euclidean):
        super().__init__(vector_euclidean)