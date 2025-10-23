from src.manifolds.isometrized_euclidean.image import l2IsometrizedImageEuclidean
from src.manifolds.isometrized_geodesic_submanifold_euclidean import l2IsometrizedGeodesicSubmanifoldEuclidean  

class l2IsometrizedImageGeodesicSubmanifoldEuclidean(l2IsometrizedGeodesicSubmanifoldEuclidean):
    def __init__(self, image_geo_sub_euclidean, num_intervals=10):
        super().__init__(l2IsometrizedImageEuclidean(image_geo_sub_euclidean, num_intervals=num_intervals))