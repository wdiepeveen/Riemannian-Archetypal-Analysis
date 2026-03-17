import torch
from sklearn.cluster import KMeans

from src.distributions.starflows.recentered import RecenteredStarFlowDistribution
from src.distributions.stars.ellipsoid.multimodal import MultiModalEllipsoidStarDistribution


class TrimmedEllipsoidStarFlowTraining(torch.nn.Module):
    def __init__(self, d, data, n_clusters, c=4/3, p=0.95, trimmed=True, cov_reg=1e-6):
        super(TrimmedEllipsoidStarFlowTraining, self).__init__()

        self.d = d
        self.shape = data.shape[1:]
        self.N = data.shape[0]
        self.data = data

        self.K = n_clusters
        
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
    def star_center(self):
        return self._star_center.reshape(*self.shape)

    def solve(self):
        # Step 1: K-means clustering
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.data.reshape(self.N, -1).numpy())
        self.cluster_labels = torch.from_numpy(kmeans.labels_).to(self.data.dtype)
        self._cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.data.dtype)

        # Step 2: Compute cluster covariances
        self._cluster_covariances = torch.zeros((self.K, self.d, self.d), dtype=self.data.dtype)
        self.cluster_sizes = []
        for k in range(self.K):
            cluster_data = self.data.reshape(self.N, -1)[self.cluster_labels == k]
            cluster_center = self._cluster_centers[k]
            cluster_size = cluster_data.shape[0]
            self._cluster_covariances[k] = torch.einsum('ni,nj->ij', cluster_data - cluster_center, cluster_data - cluster_center) / (cluster_size - 1) + self.cov_reg * torch.eye(self.d, dtype=self.data.dtype)
            self.cluster_sizes.append(cluster_size)

        self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)
        self.weights = torch.tensor(self.cluster_sizes, dtype=self.data.dtype) / self.N

        # Step 3: Compute star center
        A = (self.weights[:, None, None] * self._inv_cluster_covariances).sum(dim=0)
        b = (self.weights[:, None] * torch.einsum('nij,nj->ni', self._inv_cluster_covariances, self._cluster_centers)).sum(dim=0)
        
        self._star_center = torch.linalg.solve(A, b)

        # Step 4: Construct starflow distribution
        self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
                                                 mus=[center - self._star_center for center in self._cluster_centers],
                                                 c=self.c, p=self.p, trimmed=self.trimmed, 
                                                 aggregation='softmax')
        self.starflow = RecenteredStarFlowDistribution(self.star_center, self.star)

        