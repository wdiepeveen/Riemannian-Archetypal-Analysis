import torch
from sklearn.mixture import GaussianMixture

from src.distributions.stars.ellipsoid.multimodal import MultiModalEllipsoidStarDistribution

class StarTraining:
    def __init__(self, shape, n_clusters, covariance_type='diag', c=4/3, p=0.95, cov_reg=1e-6):
        super(StarTraining, self).__init__()

        self.d = torch.prod(torch.tensor(shape)).item()
        self.shape = shape

        self.K = n_clusters
        self.covariance_type = covariance_type
        
        self.c = c
        self.p = p
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

    def fit(self, data, labels=None):
        N = data.shape[0]
        assert list(data.shape[1:]) == self.shape, f"Data shape {list(data.shape[1:])} does not match expected shape {self.shape}"

        # Step 1: Gaussian Mixture clustering
        if labels is None:
            gmm = GaussianMixture(n_components=self.K, covariance_type=self.covariance_type, random_state=0)
            gmm.fit(data.reshape(N, -1).numpy())
            labels = torch.from_numpy(gmm.predict(data.reshape(N, -1).numpy())).to(data.dtype)
            # cluster_centers_ = torch.from_numpy(gmm.means_).to(data.dtype)

        # Step 2: Compute cluster covariances
        cluster_centers = []
        cluster_covariances = []
        cluster_sizes = []
        for k in range(self.K):
            cluster_data = data.reshape(N, -1)[labels == k]
            cluster_size = cluster_data.shape[0]
            if cluster_size > self.d:  # Ensure we have enough data points to compute covariance
                cluster_sizes.append(cluster_size)

                cluster_mu = cluster_data.mean(dim=0)
                cluster_centers.append(cluster_mu)

                cluster_cov = cluster_data.T.cov() + self.cov_reg * torch.eye(self.d, dtype=data.dtype)  # Regularize covariance
                cluster_covariances.append(cluster_cov)
                
            else: 
                self.K -= 1

        self._cluster_covariances = torch.stack(cluster_covariances)
        self._cluster_centers = torch.stack(cluster_centers)
        self.cluster_sizes = torch.tensor(cluster_sizes, dtype=data.dtype)
        self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)

        # Step 3: Construct starflow distribution
        self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
                                                        mus=self._cluster_centers,
                                                        c=self.c, p=self.p, 
                                                        trimmed=False, aggregation='softmax')

        