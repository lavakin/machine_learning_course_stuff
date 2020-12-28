#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    print(xs)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []
    for order in range(1, args.range + 1):
        # Create features of x^1, ..., x^order.
        features = np.transpose(np.asarray([[ x**n for x in xs ] for n in range (1,order + 1)]))
        data_train, data_test, target_train, target_test = train_test_split(features, ys, test_size=args.test_size, random_state=args.seed)
        reg = LinearRegression().fit(data_train, target_train)
        prediction = reg.predict(data_test)
        rmse = np.sqrt(mean_squared_error(prediction, target_test))
        rmses.append(rmse)

    return rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))

