#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def calculate_weights(input_data, target_data):
    transposed = input_data.transpose()
    inversed = (np.linalg.inv(np.matmul(transposed, input_data)))
    weights = np.matmul(np.matmul(inversed, transposed), target_data)    
    return weights


# Compute root mean square error on the test set predictions
def get_rmse(test_data, target_test_data, weights):
    #Predict target values on the test set
    prediction = np.matmul(test_data, weights)
    error = np.subtract(prediction, target_test_data)
    MSE = sum([x**2 for x in error])/len(error)
    RMSE = np.sqrt(MSE)
    return RMSE


def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    dataset.data  = np.asarray([np.append(x,1) for x in dataset.data])
    data_train, data_test, data_target_train, data_target_test = train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)
    weights = calculate_weights(data_train, data_target_train) 
    rmse = get_rmse(data_test, data_target_test, weights)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
