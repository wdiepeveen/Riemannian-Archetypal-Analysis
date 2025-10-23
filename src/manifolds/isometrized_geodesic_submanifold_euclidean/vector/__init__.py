from src.manifolds.isometrized_euclidean.vector import l2IsometrizedVectorEuclidean
from src.manifolds.isometrized_geodesic_submanifold_euclidean import l2IsometrizedGeodesicSubmanifoldEuclidean  

class l2IsometrizedVectorGeodesicSubmanifoldEuclidean(l2IsometrizedGeodesicSubmanifoldEuclidean):
    def __init__(self, vector_geo_sub_euclidean, num_intervals=10):
        super().__init__(l2IsometrizedVectorEuclidean(vector_geo_sub_euclidean, num_intervals=num_intervals))