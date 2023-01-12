import pprint
import torch
from torch import nn
from torch import distributed as dist
from .. import utils

pairwise = {'dot': utils.dot_product, 'cosine': utils.cosine_similarity, 'euclidean': utils.euclidean_distance}


class Base(nn.Module):
    def __init__(self, algorithm: str, pairwise_metric: str):
        super().__init__()
        assert utils.is_distributed(), print("Algorithm created before distributed initialization. ")
        self.algorithm = algorithm
        self.pairwise_metric = pairwise_metric
        if utils.is_distributed():
            self.world_size = 1
            self.rank = 0
            self.distributed = False
        else:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.distributed = True
        self._results = None

    def pairwise_similarity(self, a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
        """
        Compute the pairwise similarity of two tensors.
        """
        M = pairwise[self.pairwise_metric](a, b, transpose_b)
        if 'euclidean' in self.pairwise_metric:
            return -M
        else:
            return M

    def pairwise_distance(self, a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
        """
        Compute the pairwise similarity of two tensors.
        """
        M = pairwise[self.pairwise_metric](a, b, transpose_b)
        if 'euclidean' in self.pairwise_metric:
            return M
        else:
            return -M

    def fit(self, X: torch.Tensor, y: torch.Tensor, gather_y: bool):
        print(f"{self.algorithm} does not have a fit method. ")
        return

    def display_results(self):
        if self.rank == 0:
            print(f"Display results for {self.algo_name}")
            # Prints the nicely formatted dictionary
            pprint.pprint(self._results)

    def gather(self, X: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather tensor z from all available gpus.
        The output tensor will be sorted according to each rank and concatenated over dim.
        """
        if not self.distributed:
            # no need to gather with 1 gpu
            return X
        gather_list = [torch.empty_like(X) for _ in range(self.world_size)]
        dist.all_gather(gather_list, X)
        return torch.cat(gather_list, dim=dim)

    def reduce(self, X: torch.Tensor, op: str):
        """
        Perform all reduce operation on a given tensor.
        """
        if not self.distributed:
            # no need to reduce with 1 gpu
            return X

        # choose the reduce method
        if op == 'sum':
            dist.all_reduce(X, op=dist.ReduceOp.SUM)
        elif op == 'mean':
            dist.all_reduce(X, op=dist.ReduceOp.SUM)
            X = X / self.world_size
        elif op == 'max':
            dist.all_reduce(X, op=dist.ReduceOp.MAX)
        elif op == 'min':
            dist.all_reduce(X, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Reduce operation: {op} is not available. ")

        return X

    @property
    def requires_split(self):
        return self._requires_split

    @property
    def results(self):
        return self._results
