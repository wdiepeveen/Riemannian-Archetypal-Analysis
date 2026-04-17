import torch

from distributions.stars.ellipsoid.old_multimodal import MultiModalEllipsoidStarDistribution

class StarConstruction:
    def __init__(self, shape, c=4/3, p=0.95, trimmed=False, cov_reg=1e-6):
        super(StarConstruction, self).__init__()

        self.d = torch.prod(torch.tensor(shape)).item()
        self.shape = shape

        self.k = 0
        self.labels = []
        
        self.c = c
        self.p = p
        self.trimmed = trimmed
        self.cov_reg = cov_reg

    @property
    def cluster_centers(self):
        return self._cluster_centers.reshape(-1, *self.shape)
    
    @property
    def cluster_end_members(self):
        return self._cluster_end_members.reshape(-1, *self.shape)
    
    @property
    def cluster_covariances(self):
        return self._cluster_covariances.reshape(-1, *self.shape, *self.shape)
    
    @property
    def inv_cluster_covariances(self):
        return self._inv_cluster_covariances.reshape(-1, *self.shape, *self.shape)
    
    def fit(self, data, labels, end_members=None, min_cluster_size=None):
        N = data.shape[0]
        if min_cluster_size is None:
            min_cluster_size = self.d + 1  # minimum size to compute covariance
        assert list(data.shape[1:]) == self.shape, f"Data shape {list(data.shape[1:])} does not match expected shape {self.shape}"

        # compute cluster covariances
        cluster_centers = []
        cluster_end_members = []
        cluster_covariances = []
        cluster_sizes = []
        for k in range(labels.max().item() + 1):
            cluster_data = data.reshape(N, -1)[labels == k]
            if cluster_data.shape[0] < min_cluster_size:
                print(f"Cluster {k} has size {cluster_data.shape[0]} which is less than minimum cluster size {min_cluster_size}. Skipping this cluster.")
            else:
                if end_members is not None:
                    cluster_end_member = end_members[k].flatten()
                    cluster_mu = cluster_end_member/2
                else:
                    cluster_mu = cluster_data.mean(dim=0)
                    cluster_end_member = 2 * cluster_mu 

                cluster_size = cluster_data.shape[0]
                cluster_sizes.append(cluster_size)

                cluster_end_members.append(cluster_end_member)
                cluster_centers.append(cluster_mu)

                cluster_cov = torch.einsum("ij,ik->jk", cluster_data - cluster_mu, cluster_data - cluster_mu) / cluster_size + self.cov_reg * torch.eye(self.d, dtype=data.dtype)  # regularize covariance
                cluster_covariances.append(cluster_cov)
                
                self.labels.append(k)
                self.k += 1

        self._cluster_centers = torch.stack(cluster_centers)
        self._cluster_end_members = torch.stack(cluster_end_members)
        self._cluster_covariances = torch.stack(cluster_covariances)
        self.cluster_sizes = torch.tensor(cluster_sizes, dtype=data.dtype)
        self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)

        # construct starflow distribution
        self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
                                                        mus=self._cluster_centers,
                                                        c=self.c, p=self.p, 
                                                        trimmed=self.trimmed, aggregation='softmax')

    # def fit(self, data, labels, min_cluster_size=None):
    #     N = data.shape[0]
    #     if min_cluster_size is None:
    #         min_cluster_size = self.d + 1  # minimum size to compute covariance
    #     assert list(data.shape[1:]) == self.shape, f"Data shape {list(data.shape[1:])} does not match expected shape {self.shape}"

    #     # compute cluster covariances
    #     cluster_centers = []
    #     cluster_covariances = []
    #     cluster_sizes = []
    #     for k in range(labels.max().item() + 1):
    #         cluster_data = data.reshape(N, -1)[labels == k]
    #         cluster_size = cluster_data.shape[0]
    #         if cluster_size >= min_cluster_size:  # ensure we have enough data points to compute covariance
    #             cluster_sizes.append(cluster_size)

    #             cluster_mu = cluster_data.mean(dim=0)
    #             cluster_centers.append(cluster_mu)

    #             cluster_cov = cluster_data.T.cov() + self.cov_reg * torch.eye(self.d, dtype=data.dtype)  # regularize covariance
    #             cluster_covariances.append(cluster_cov)
    #             self.k += 1
    #             self.labels.append(k)

    #     self._cluster_covariances = torch.stack(cluster_covariances)
    #     self._cluster_centers = torch.stack(cluster_centers)
    #     self.cluster_sizes = torch.tensor(cluster_sizes, dtype=data.dtype)
    #     self._inv_cluster_covariances = torch.cat([torch.linalg.inv(cov)[None] for cov in self._cluster_covariances], dim=0)

    #     # construct starflow distribution
    #     self.star = MultiModalEllipsoidStarDistribution(covs=self._cluster_covariances, 
    #                                                     mus=self._cluster_centers,
    #                                                     c=self.c, p=self.p, 
    #                                                     trimmed=self.trimmed, aggregation='softmax')

        