import torch

from src.clustering import Clustering

class KMeans(Clustering):
    
    def __init__(self, euclidean, num_clusters=3, tol=1e-4, max_iter=100, bary_tol=1e-1, bary_max_iter=10, bary_step_size=1., bary_red_coef=0.5): # TODO add step size, max iter and tolerance for barycentre
        super().__init__(euclidean)
        self.num_clusters = num_clusters
        self.tol = tol
        self.max_iter = max_iter
        
        self.bary_tol = bary_tol
        self.bary_max_iter = bary_max_iter
        self.bary_step_size = bary_step_size
        self.bary_red_coef = bary_red_coef

        self.centroids = None
        self.labels_ = None

    def fit(self, x):
        """
        Fit the KMeans model to the data.
        
        :param x: N x [Epoint]
        """
        with torch.no_grad():
            # Initialize centroids randomly from the data points
            if self.centroids is None:
                indices = torch.randperm(x.size(0))[:self.num_clusters]
                centroids = x[indices]
            else:
                centroids = self.centroids

            distances = torch.zeros((x.size(0), self.num_clusters), device=x.device)
            for i in range(self.max_iter):
                # Assign clusters based on closest centroid
                for j in range(self.num_clusters):
                    distances[:,j] = self.euclidean.distance(x[None], centroids[j][None,None])[0,:,0]
                labels = torch.argmin(distances, dim=1)

                # Compute new centroids
                new_centroids = torch.stack([self.euclidean.barycentre(x[labels == j], tol=self.bary_tol, max_iter=self.bary_max_iter, step_size=self.bary_step_size, red_coef=self.bary_red_coef) for j in range(self.num_clusters)])

                # Check for convergence
                if torch.norm(new_centroids - centroids) < self.tol:
                    print(f"Converged after {i+1} iterations")
                    break

                centroids = new_centroids

            self.centroids = centroids
            self.labels_ = labels

    def predict(self, x):
        """
        Predict the closest cluster for each point in x.
        
        :param x: N x [Epoint]
        :return: N tensor of cluster labels
        """
        with torch.no_grad():
            distances = self.euclidean.distance(x[:, None], self.centroids[None, :])
            return torch.argmin(distances, dim=1)