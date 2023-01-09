import torch
from torch import nn
from torch import distributed as dist
from .. import utils

pairwise = {'dot': utils.dot_product, 'cosine': utils.cosine_similarity, 'euclidean': utils.euclidean_distance}


class BaseAlgorithm(nn.Module):
    def __init__(self, algorithm: str, pairwise_metric: str):
        super().__init__()
        assert utils.is_distributed(), print("Algorithm created before distributed initialization. ")
        self.algorithm = algorithm
        self.pairwise_metric = pairwise_metric
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def gather(self, z: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather tensor z from all available gpus.
        The output tensor will be sorted according to each rank and concatenated over dim.
        """
        gather_list = [torch.empty_like(z) for _ in range(self.world_size)]
        dist.all_gather(gather_list, z)
        return torch.cat(gather_list, dim=dim)

    def pairwise_similarity(self, a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
        """
        Compute the pairwise similarity of two tensors
        """
        M = pairwise[self.pairwise_metric](a, b, transpose_b)
        if 'euclidean' in self.pairwise_metric:
            return -M
        else:
            return M



