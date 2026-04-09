import torch

from src.dimension_reduction.convex_simplex_structured_matrix_factorization import ConvexSimplexStructuredMatrixFactorizationSolver
from src.distributions.stars.ellipsoid.multimodal import MultiModalEllipsoidStarDistribution

class StarTraining:
    def __init__(self, shape, n_clusters, N_max=-1, c=4/3, p=0.95, trimmed=False, cov_reg=1e-6):
        super(StarTraining, self).__init__()

        self.d = torch.prod(torch.tensor(shape)).item()
        self.shape = shape

        self.r = n_clusters
        self.N_max = N_max
        self.k = n_clusters
        
        self.c = c
        self.p = p
        self.trimmed = trimmed
        self.cov_reg = cov_reg

    @property
    def cluster_centers(self):
        return self._cluster_centers.reshape(-1, *self.shape)
    
    @property
    def cluster_covariances(self):
        return self._cluster_covariances.reshape(-1, *self.shape, *self.shape)
    
    @property
    def inv_cluster_covariances(self):
        return self._inv_cluster_covariances.reshape(-1, *self.shape, *self.shape)
    
    @property
    def end_members(self):
        return self._end_members.reshape(-1, *self.shape)

    def fit(self, data, labels=None, min_cluster_size=None):
        N = data.shape[0]
        if min_cluster_size is None:
            min_cluster_size = self.d + 1  # Minimum size to compute covariance
        assert list(data.shape[1:]) == self.shape, f"Data shape {list(data.shape[1:])} does not match expected shape {self.shape}"

        # Step 1: CSSMF clustering
        if labels is None:
            X = data.reshape(N, -1).T  # use the generated data as input
            solver = ConvexSimplexStructuredMatrixFactorizationSolver(d=X.shape[0], N=X.shape[1])
            solver.fit(X, r=self.r, N_max=self.N_max, max_iter=500)
            self._end_members = solver.V.T
            labels = solver.labels.cpu()

        # Step 2: Compute cluster covariances
        cluster_centers = []
        cluster_covariances = []
        cluster_sizes = []
        for k in range(self.r):
            cluster_data = data.reshape(N, -1)[labels == k]
            cluster_size = cluster_data.shape[0]
            if cluster_size >= min_cluster_size:  # Ensure we have enough data points to compute covariance
                cluster_sizes.append(cluster_size)

                cluster_mu = cluster_data.mean(dim=0)
                cluster_centers.append(cluster_mu)

                cluster_cov = cluster_data.T.cov() + self.cov_reg * torch.eye(self.d, dtype=data.dtype)  # Regularize covariance
                cluster_covariances.append(cluster_cov)
                
            else: 
                self.k -= 1

        self._cluster_covariances = torch.stack(cluster_covariances)
        self._cluster_centers = torch.stack(cluster_centers)
        self.cluster_sizes = torch.tensor(cluster_sizes, dtype=data.dtype)
        self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)

        # Step 3: Construct starflow distribution
        self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
                                                        mus=self._cluster_centers,
                                                        c=self.c, p=self.p, 
                                                        trimmed=self.trimmed, aggregation='softmax')

        