import torch
import math
from base_cluster import BaseCluster


class KMeans(BaseCluster):
    def __init__(self, algorithm: str, pairwise_metric: str, num_classes: int, num_iters: int,
                 init_method: str = 'random'):
        super().__init__(algorithm, pairwise_metric, num_classes, num_iters)
        self.algo_name = f"{self.num_classes}-means clustering"
        self.init_method = init_method
        self._requires_split = False
        self._centroids = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform KMeans clustering on a 2d tensor.
        """

        # centroids initialization
        centroids = self.init_centroids(X=X)

        assignments = None
        for _iter in range(self.num_iters):
            # distance to centroids
            D = self.pairwise_distance(X, centroids)
            # nearest centroid
            assignments = D.argmin(dim=-1)

            # create new centroids
            for c in torch.unique(assignments):
                centroids[c] = X[assignments == c, :].mean(dim=0)

            # sync
            centroids = self.reduce(X=centroids, op="mean")

        self._centroids = centroids
        self._assignments = assignments

    def init_centroids(self, X: torch.Tensor):
        """
        Initialize centroids
        """
        if self.init_method == "random":
            # initialize centroids randomly
            centroids = torch.randn((self.num_classes, X.shape[1])).cuda(non_blocking=True)

        elif self.init_method == "random_assign":
            # randomly assign each sample to a centroid and compute its mean.
            # the samples will distribute uniformly into the clusters
            n_samples = X.shape[0]
            n_in_class = int(math.ceil(n_samples / self.num_classes))
            assignments = torch.arange(self.num_classes).repeat(n_in_class)[:n_samples]
            # shuffle assignments
            assignments = assignments[torch.randperm(len(assignments))]

            # create centroids from random assignments
            centroids = torch.randn((self.num_classes, X.shape[1])).cuda(non_blocking=True)
            # create new centroids
            for c in range(self.num_classes):
                centroids[c] = X[assignments == c, :].mean(dim=0)

        else:
            raise ValueError(f"Init method {self.init_method} is unavailable. ")

        # sync
        centroids = self.reduce(X=centroids, op="mean")
        return centroids

    @property
    def centroids(self):
        return self._centroids









