import torch
from ..base import Base


class KNN(Base):

    def __init__(self, algorithm: str, pairwise_metric: str, batch_size: int, k: int,
                 num_classes: int, temperature: float):
        super().__init__(algorithm, pairwise_metric)
        self.k = k
        self.algo_name = f"{k}-nn Classification"
        self.temperature = temperature
        self.num_classes = num_classes
        self.batch_size = batch_size
        self._requires_split = True
        assert self.k > 0 and self.num_classes > 1 and self.temperature > 0 and self.batch_size > 0

    def fit(self, X: torch.Tensor, y: torch.Tensor, gather_y: bool):
        """
        Fit train data into module
        :param X: Train features of each gpu.
        :param y: Labels of each gpu if gather_y is True. Labels gathered from all gpus if gather_y is False.
        :param gather_y: The labels needs to gathered from all gpus. Set to True if y including local labels.
        """
        self._X = X
        if 'euclidean' not in self.pairwise_metric:
            self._X = self._X.t()   # save the transpose

        if gather_y:
            self._y = self.gather(y, dim=0).view(1, -1)
        else:
            self._y = y

    def forward(self, X: torch.Tensor):
        """
        Perform KNN classification.
        """
        num_test_samples = X.shape[0]
        rank_offset = self._y.shape[1] // self.world_size
        retrieval_one_hot = torch.zeros(self.k, self.num_classes).cuda(non_blocking=True)

        # classification loop
        self._assignments = []
        for idx in range(0, X.shape[0], self.batch_size):
            z = X[idx: min((idx + self.batch_size), num_test_samples), :]
            batch_size = z.shape[0]

            # --- find the k-most similar local training samples --- #
            similarity = self.pairwise_similarity(z, self._X, transpose_b=False)
            sim_k, indices = similarity.topk(self.k, largest=True, sorted=True)

            # --- gather top-k information from all gpus --- #
            # TODO: should be done with one sync
            sim_k_all = self.gather(sim_k)
            indices_all = self.gather(indices + self.rank * rank_offset)

            # --- find the real top-k train samples from gathered information --- #
            sim_k, rank_indices = sim_k_all.topk(self.k, largest=True, sorted=True)
            indices = torch.gather(indices_all, 1, rank_indices)

            candidates = self.y.expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * self.k, self.num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.type(torch.int64).view(-1, 1), 1)

            sim_transform = sim_k.clone().div_(self.temperature).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, self.num_classes),
                                        sim_transform.view(batch_size, -1, 1), ), 1, )

            # --- find predictions that match the target --- #
            _, assignments = probs.sort(1, True)
            self._assignments.append(assignments)

        self._assignments = torch.stack(self._assignments)

    def compute_results(self, y) -> dict:
        """
        Return the top 1 and top 5 knn accuracy.
        """
        assert self._assignments is not None, print("Error: You need to call forward method first. ")
        assert len(y) == len(self._assignments)

        results = {"top_1": 0., "top_5": 0.}
        i = 0
        total = 0
        num_test_samples = y.shape[0]
        for idx in range(0, y.shape[0], self.batch_size):
            targets = y[idx: min((idx + self.batch_size), num_test_samples)]
            assignments = self._assignments[i]
            correct = assignments.eq(targets.data.view(-1, 1))

            results["top_1"] += correct.narrow(1, 0, 1).sum().item()
            results["top_5"] += correct.narrow(1, 0, min(5, self.k)).sum().item()  # top5 does not make sense if k < 5

            total += len(targets)
            i += 1

        results["top_1"] = results["top_1"] * 100.0 / total
        results["top_5"] = results["top_5"] * 100.0 / total
        self._results = results
        return results

    def display(self):
        print(f"Display results for {self.k}-KNN classification: ")
        print(f"Top 1 accuracy: {self._results['top_1']:.4f}%")
        print(f"Top 5 accuracy: {self._results['top_5']:.4f}%")








