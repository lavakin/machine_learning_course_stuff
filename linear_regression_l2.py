#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")


def calculate_rmse(alpha):
    return 


def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    data_train, data_test, target_train, target_test = train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    lambdas = np.geomspace(0.01, 100, num=500)
    best_rmse, best_lambda = min((np.sqrt(mean_squared_error(Ridge(alpha=x).fit(data_train, target_train).predict(data_test), target_test)), x) for x in lambdas)


    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
