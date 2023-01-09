import torch
import argparse
import utils


def parse_args():
    parser = argparse.ArgumentParser('main', add_help=False)

    # Global args
    parser.add_argument('--algorithm', default='knn', type=str, choices=['knn'])
    parser.add_argument('--pairwise_metric', default='dot', type=str, choices=['dot', 'cosine', 'euclidean'],
                        help=""" The metric to compute the pairwise similarity/distance matrix. """)
    parser.add_argument('--num_classes', default=None, type=int,
                        help=""" Number of classes in the dataset. Only applicable to classification/clustering. """)
    parser.add_argument('--batch_size', default=None, type=int,
                        help=""" Process the algorithm in batches of this size (memory efficient). """)
    parser.add_argument('--path_train_features', default='./data/train/', type=str,
                        help=""" Path to train samples. If split_to_gpus each gpu should get a subset of this. """)
    parser.add_argument('--path_train_labels', default='./data/train/', type=str,
                        help=""" Path to train labels. If split_to_gpus each gpu should get a subset of this. """)
    parser.add_argument('--path_test_features', default=None, type=int,
                        help=""" Path to test samples. They should be equal for all gpus. """)
    parser.add_argument('--path_test_labels', default=None, type=int,
                        help=""" Path to test labels. They should be equal for all gpus. """)
    parser.add_argument('--split_to_gpus', default=False, type=utils.bool_flag,
                        help=""" If true, split train data among gpus. """)
    parser.add_argument('--shuffle', default=False, type=utils.bool_flag,
                        help=""" If true, shuffle data before splitting. """)

    # KNN args
    parser.add_argument('--k', default=20, type=int, help="""Number of neighbours. """)
    parser.add_argument('--temperature', default=1.,
                        type=float, help="""Divide the similarities before making them probabilities. """)

    # Misc
    parser.add_argument('--seed', default=0, type=int, help="""Random seed.""")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="""Please ignore and do not set this argument.""")
    return parser.parse_args()


def main():
    """ Example of how the simple pipeline should look like """

    args = parse_args()

    utils.is_distributed()
    utils.seed_all(seed=args.seed)

    # get the chosen algorithm
    algo = utils.get_algorithm(args=args)
    X_train, y_train, X_test, y_test = utils.load_data(args)

    algo(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main()
