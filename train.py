import numpy as np
import pandas as pd
from preproc import *
import argparse
from RF import RandomForestClassifier
import pickle as pcl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='resources/train.csv', help='train dataset')
    parser.add_argument('--model', '-m', default='model.pcl', help='file to save model into')
    parser.add_argument('--silent', '-s', default=False, help='silent mode', action='store_true')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset, index_col=0)

    delta = df.iloc[:, 0]
    Y = df.iloc[:, 1:401].values
    X = df.iloc[:, 401:].values.reshape(-1, 20, 20)

    kernel_size = 3
    model_map = dict()
    for i in range(1, 6):
        if not args.silent:
            print(f"Fitting model for delta={i}")
        X_i = X[delta == i]
        Y_i = Y[delta == i]

        X_train, Y_train = prepare_data(X_i, Y_i, kernel_size=kernel_size)
        X_train = X_train.astype(np.float64)
        Y_train = Y_train.astype(np.float64)

        rf = RandomForestClassifier(verbose=not args.silent)
        rf.fit(X_train, Y_train)
        model_map[i] = {'trees': rf.trees, 'feat_ids_by_tree': rf.feat_ids_by_tree, 'unique_labels': rf.unique_labels}
    
    with open(args.model, 'wb') as f:
        pcl.dump(model_map, f)
        if not args.silent:
            print(f"Model saved into {args.model}")