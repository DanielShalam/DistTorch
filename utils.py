import os
import argparse

import torch
from torch import nn
from torch import distributed as dist
import numpy as np
from algorithms.classification.knn import KNN


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def init_distributed(args):
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if 'SLURM_PROCID' in os.environ or args.is_slurm_job:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
        print(f'Initialize SLURM job, rank {args.rank}, gpu {args.gpu}, world size {args.world_size}. ')

    # launched with torch.distributed.launch
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Support GPU Only.')
        exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}, gpu {}): {}'.format(args.rank, args.gpu, args.dist_url), flush=True)
    dist.barrier()


def get_algorithm(args):
    """
    Get the algorithm module.
    """
    if 'knn' in args.algorithm:
        return KNN(algorithm=args.algorithm, pairwise_metric=args.pairwise_metric, k=args.k,
                   batch_size=args.batch_size, num_classes=args.num_classes, temperature=args.temperature)
    if 'kmeans' in args.algorithm:
        return KNN(algorithm=args.algorithm, pairwise_metric=args.pairwise_metric, k=args.k,
                   batch_size=args.batch_size, num_classes=args.num_classes, temperature=args.temperature)
    else:
        print(f"Error: {args.algorithm} unavailable unavailable for now. ")
        exit()


def load_data(args, requires_split: bool):
    """
    Load data. Only support .py files for now.
    """
    suffix = args.path_train_features.split('.')[-1]
    if suffix == 'pt':
        train_features = torch.load(args.path_train_features).cuda(non_blocking=True)
        train_labels = torch.load(args.path_train_labels).cuda(non_blocking=True)

        # split train data among gpus
        if args.split_to_gpus:
            train_features, train_labels = split_to_gpus(train_features, train_labels, shuffle=args.shuffle)

        if not requires_split:
            print(
                f"Data loaded successfully with {train_features.shape[0]} local samples of dim {train_features.shape[1]}")
            return train_features, train_labels

        test_features = test_labels = None
        if args.path_test is not None:
            test_features = torch.load(args.path_test_features).cuda(non_blocking=True)
            test_labels = torch.load(args.path_test_labels).cuda(non_blocking=True)

        print(f"Data loaded successfully with {train_features.shape[0]} local samples of dim {train_features.shape[1]}")
        return train_features, train_labels, test_features, test_labels

    else:
        print(f"Error: Data loading for datatype {suffix} is unavailable for now. ")
        exit()


def split_to_gpus(train_features, train_labels, shuffle: bool):
    """
    Split data among gpus.
    """
    num_samples = train_features.shape[0]
    if shuffle:
        indexes_all = torch.randperm(n=num_samples)
    else:
        indexes_all = torch.arange(num_samples)

    # set the indexes of each gpu
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    indexes_per_gpu = indexes_all.chunk(world_size)
    assert world_size == len(indexes_per_gpu)
    # we need to make sure that all workers get the same number of samples
    min_samples_per_gpu = min([len(indexes) for indexes in indexes_per_gpu])

    train_features_local = train_features[indexes_per_gpu[rank, :min_samples_per_gpu]]
    train_labels_local = None
    if train_labels is not None:
        train_labels_local = train_labels[indexes_per_gpu[rank, :min_samples_per_gpu]]

    return train_features_local, train_labels_local


def seed_all(seed: int = 0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def dot_product(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True):
    """
    Compute the pairwise dot product between two tensors
    """
    if transpose_b:
        return torch.matmul(a, b.t())

    return torch.matmul(a, b)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True):
    """
    Compute the pairwise cosine-similarity between two tensors
    """
    if transpose_b:
        return torch.matmul(a, b.t())

    return torch.matmul(a, b)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = None):
    """
    Compute the pairwise euclidean distance between two tensors
    """
    return torch.cdist(a, b, p=2)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
