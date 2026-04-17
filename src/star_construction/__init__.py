import archetypes
import torch
from archetypes import AA

from src.distributions.stars.ellipsoid.old_multimodal import MultiModalEllipsoidStarDistribution

class StarConstruction:
    def __init__(self, n_clusters=None, c=4/3, p=0.95, trimmed=False, cov_reg=1e-6, n_archetypes=None):
        assert n_clusters > 0, "Number of clusters must be positive."
        assert n_archetypes > 0, "Number of archetypes must be positive."

        self.n_clusters = n_clusters
        
        self.c = c
        self.p = p
        self.trimmed = trimmed
        self.cov_reg = cov_reg

        self.n_archetypes = n_archetypes
    
    def fit(self, data, labels):
        """
        
        :param data: N x d tensor
        :param labels: N list
        """
        assert len(labels) == data.shape[0], "Number of labels must match number of data points."
        assert len(torch.unique(labels)) <= self.n_clusters, "Number of unique labels must be less than or equal to n_clusters."
        # else:
            # # do classification based on Archetypal Analysis -- TODO remove because this has to be done on the log space of the barycentre of the archetypes
            # aa = AA(self.n_clusters, init='furthest_sum')
            # aa.fit(data)
            # labels = torch.tensor(aa.labels_)

        # construct the star distribution and archetypes for each cluster + compute the end members of the star distribution for each cluster
        unique_clusters = torch.unique(labels)
        cluster_centers = []
        cluster_covariances = []
        archetyepes = []
        for label in unique_clusters:
            cluster_data = data[labels == label]
            cluster_size = cluster_data.shape[0]
            
            # star components for the cluster
            cluster_center = cluster_data.mean(dim=0)
            cluster_covariance = torch.einsum("ij,ik->jk", cluster_data - cluster_center, cluster_data - cluster_center) / cluster_size  + self.cov_reg * torch.eye(cluster_data.shape[1])
            
            cluster_centers.append(cluster_center)
            cluster_covariances.append(cluster_covariance)

            # archetypes for the cluster -- TODO this should be done on the log space of the barycentre of the archetypes
            aa_label = AA(self.n_archetypes+1, init='furthest_sum')
            aa_label.fit(cluster_data)
            cluster_archetypes = torch.from_numpy(aa_label.archetypes_)
            norms = torch.norm(cluster_archetypes, dim=1)
            _, idx_top = torch.topk(norms, k=self.n_archetypes, largest=True)
            cluster_archetypes = cluster_archetypes[idx_top]
            archetyepes.append(cluster_archetypes)

        # construct the joint star distribution from the clusters
        self.cluster_centers = torch.stack(cluster_centers)
        self.cluster_covariances = torch.stack(cluster_covariances)
        self.star = MultiModalEllipsoidStarDistribution(covs=self.cluster_covariances, 
                                                        mus=self.cluster_centers,
                                                        c=self.c, p=self.p, 
                                                        trimmed=self.trimmed, aggregation='softmax')

        # construct the joint archetypes from the cluster archetypes
        self.archetypes = torch.cat(archetyepes, dim=0)