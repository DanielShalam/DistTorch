import torch
from ..base import Base
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class BaseCluster(Base):
    def __init__(self, algorithm: str, pairwise_metric: str, num_classes: int, num_iters: int):
        super().__init__(algorithm, pairwise_metric)
        self._num_classes = num_classes
        self._num_iters = num_iters
        self._assignments = None

    def compute_results(self, y) -> dict:
        """ Return clustering results from pseudo-assignments. """
        assert self._assignments is not None, print("Error: You need to call forward method first. ")
        assert len(y) == len(self._assignments)

        np_y = y.cpu().numpy()

        # compute results
        np_assign = self._assignments.cpu().numpy()
        nmi = normalized_mutual_info_score(labels_true=np_y, labels_pred=np_assign)
        ari = adjusted_rand_score(labels_true=np_y, labels_pred=np_assign)

        self._results = {"NMI": nmi, "ARI": ari}

    def forward(self, X: torch.Tensor):
        pass

    @property
    def assignments(self):
        return self._assignments

    @property
    def num_iters(self):
        return self._num_iters

    @property
    def num_classes(self):
        return self._num_classes





