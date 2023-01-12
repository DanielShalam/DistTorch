import argparse
import utils


def parse_args():
    parser = argparse.ArgumentParser('main', add_help=False)

    # Global args
    parser.add_argument('--algorithm', default='knn', type=str, choices=['knn', 'kmeans'])
    parser.add_argument('--pairwise_metric', default='dot', type=str, choices=['dot', 'cosine', 'euclidean'],
                        help="""The metric to compute the pairwise similarity/distance matrix. """)
    parser.add_argument('--num_classes', default=None, type=int,
                        help="""Number of classes in the dataset. Only applicable to classification/clustering. """)
    parser.add_argument('--batch_size', default=None, type=int,
                        help="""Process the algorithm in batches of this size (memory efficient). """)
    parser.add_argument('--path_train_features', default='./data/train/', type=str,
                        help="""Path to train samples. If split_to_gpus each gpu should get a subset of this. """)
    parser.add_argument('--path_train_labels', default='./data/train/', type=str,
                        help="""Path to train labels. If split_to_gpus each gpu should get a subset of this. """)
    parser.add_argument('--path_test_features', default=None, type=int,
                        help="""Path to test samples. Leave None if algorithm not requires 2 sets. """)
    parser.add_argument('--path_test_labels', default=None, type=int,
                        help="""Path to test labels. Leave None if algorithm not requires 2 sets. """)
    parser.add_argument('--split_to_gpus', default=False, type=utils.bool_flag,
                        help="""If true, split train data among gpus. """)
    parser.add_argument('--shuffle', default=False, type=utils.bool_flag,
                        help="""If true, shuffle data before splitting. """)

    # Method specific args
    parser.add_argument('--k', default=20, type=int,
                        help="""(KNN) Number of neighbours. """)
    parser.add_argument('--temperature', default=1., type=float,
                        help="""(KNN) Divide the similarities before making them probabilities. """)
    parser.add_argument('--init_method', default=20, type=int, choices=["random", "uniform_assign"],
                        help="""(K-means) Centroids initialization. """)

    # Misc
    parser.add_argument('--seed', default=0, type=int, help="""Random seed.""")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="""Please ignore and do not set this argument.""")
    return parser.parse_args()


def main():
    """ Example script """
    args = parse_args()

    # initialize DDP
    utils.init_distributed(args=args)
    utils.seed_all(seed=args.seed)

    # define the algorithm
    dist_algorithm = utils.get_algorithm(args=args)

    # load data, some algorithms requires a train/test split.
    # we can use the 'requires_split' attribute of each algorithm to differ between them
    if dist_algorithm.requires_split:
        # case the algorithm need both train/test splits.
        X_train, y_train, X, y = utils.load_data(args, requires_split=dist_algorithm.requires_split)
        # fit the algorithm with train data
        dist_algorithm.fit(X=X_train, y=y_train, gather_y=True)
    else:
        # the function will return X and y only for train paths supplied in args
        X, y = utils.load_data(args, requires_split=dist_algorithm.requires_split)
        # # note that calling the fit function here will do nothing, so that is still ok
        # # dist_algorithm.fit(X=X, y=y, gather_y=True)

    # run the algorithm
    dist_algorithm(X=X)

    # compute results
    results = dist_algorithm.compute_results(y=y)
    # and display them nicely
    dist_algorithm.display()


if __name__ == '__main__':
    main()
