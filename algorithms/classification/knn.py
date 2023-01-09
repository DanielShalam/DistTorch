import torch
from .. import base


class KNN(base.BaseAlgorithm):

    def __init__(self, algorithm: str, pairwise_metric: str, batch_size: int, k: int,
                 num_classes: int, temperature: float):
        super().__init__(algorithm, pairwise_metric)
        self.k = k
        self.temperature = temperature
        self.num_classes = num_classes
        self.batch_size = batch_size
        assert self.k > 0 and self.num_classes > 1 and self.temperature > 0 and self.batch_size > 0

    def forward(self, X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
                gather_y_train: bool):
        """
        Running KNN classification.
        y_train needs to include labels from all gpus. In the format as if you called self.gather(y_train).
        set gather_y_train to True will do that automatically.
        """

        if gather_y_train:
            y_train_gathered = self.gather(y_train, dim=0).view(1, -1)
        else:
            y_train_gathered = y_train.view(1, -1)

        rank_offset = y_train_gathered.shape[1] // self.world_size
        retrieval_one_hot = torch.zeros(self.K, self.num_classes).cuda(non_blocking=True)
        num_test_images = X_test.shape[0]

        if 'euclidean' not in self.pairwise_metric:
            X_train = X_train.t()

        top1, top5, total = 0.0, 0.0, 0

        # classification loop
        for idx in range(0, X_test.shape[0], self.batch_size):
            z = X_test[idx: min((idx + self.batch_size), num_test_images), :]
            targets = y_test[idx: min((idx + self.batch_size), num_test_images)]
            batch_size = z.shape[0]

            # --- find the k-most similar local training samples --- #
            similarity = self.pairwise_similarity(z, X_train, transpose_b=False)
            sim_k, indices = similarity.topk(self.k, largest=True, sorted=True)

            # --- gather top-k information from all gpus --- #
            # TODO: should be done with 1 gather operation
            sim_k_all = self.gather(sim_k)
            indices_all = self.gather(indices + self.rank * rank_offset)

            # --- find the real top-k train samples from gathered information --- #
            sim_k, rank_indices = sim_k_all.topk(self.k, largest=True, sorted=True)
            indices = torch.gather(indices_all, 1, rank_indices)

            candidates = y_train_gathered.expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * self.k, self.num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.type(torch.int64).view(-1, 1), 1)

            sim_transform = sim_k.clone().div_(self.temperature).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, self.num_classes),
                                        sim_transform.view(batch_size, -1, 1), ), 1, )

            # --- find predictions that match the target --- #
            _, predictions = probs.sort(1, True)
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, self.k)).sum().item()  # top5 does not make sense if k < 5
            total += batch_size

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.display_results(top1, top5)

        return top1, top5

    def display_results(self, top1: float, top5: float):
        print(f"{self.k}-KNN results: ")
        print(f"Top 1 accuracy: {top1:.4f}%")
        print(f"Top 5 accuracy: {top5:.4f}%")








