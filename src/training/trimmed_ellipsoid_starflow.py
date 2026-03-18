import torch
from sklearn.cluster import KMeans

from src.distributions.starflows.recentered import RecenteredStarFlowDistribution
from src.distributions.stars.ellipsoid.multimodal import MultiModalEllipsoidStarDistribution


class TrimmedEllipsoidStarFlowTraining(torch.nn.Module):
    def __init__(self, shape, n_clusters, c=4/3, p=0.95, trimmed=True, cov_reg=1e-6):
        super(TrimmedEllipsoidStarFlowTraining, self).__init__()

        self.d = torch.prod(torch.tensor(shape)).item()
        self.shape = shape
        # self.N = shape[0]

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

    def fit(self, data, star_center=None):
        N = data.shape[0]
        assert list(data.shape[1:]) == self.shape, f"Data shape {list(data.shape[1:])} does not match expected shape {self.shape}"

        # Step 1: K-means clustering
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(data.reshape(N, -1).numpy())
        cluster_labels = torch.from_numpy(kmeans.labels_).to(data.dtype)
        cluster_centers_ = torch.from_numpy(kmeans.cluster_centers_).to(data.dtype)

        # Step 2: Compute cluster covariances
        cluster_centers = []
        cluster_covariances = []
        cluster_sizes = []
        for k in range(self.K):
            cluster_data = data.reshape(N, -1)[cluster_labels == k]
            cluster_center = cluster_centers_[k]
            cluster_size = cluster_data.shape[0]
            if cluster_size > 1:
                cluster_centers.append(cluster_center)
                cluster_covariances.append(torch.einsum('ni,nj->ij', cluster_data - cluster_center, cluster_data - cluster_center) / (cluster_size - 1) + self.cov_reg * torch.eye(self.d, dtype=data.dtype))
                cluster_sizes.append(cluster_size)
            else: 
                self.K -= 1

        self._cluster_covariances = torch.stack(cluster_covariances)
        self._cluster_centers = torch.stack(cluster_centers)
        self.cluster_sizes = torch.tensor(cluster_sizes, dtype=data.dtype)
        self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)
        self.weights = self.cluster_sizes / N

        # Step 3: Compute star center
        if star_center is not None:
            self._star_center = star_center.reshape(-1)
        else:
            A = (self.weights[:, None, None] * self._inv_cluster_covariances).sum(dim=0)
            b = (self.weights[:, None] * torch.einsum('nij,nj->ni', self._inv_cluster_covariances, self._cluster_centers)).sum(dim=0)
            self._star_center = torch.linalg.solve(A, b)

        # Step 4: Construct starflow distribution
        self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
                                                 mus=[center - self._star_center for center in self._cluster_centers],
                                                 c=self.c, p=self.p, trimmed=self.trimmed, 
                                                 aggregation='softmax')
        self.starflow = RecenteredStarFlowDistribution(self.star_center, self.star)

        