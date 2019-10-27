import numpy as np
import pandas as pd
from preproc import *
import argparse
from RF import RandomForestClassifier
import pickle as pcl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='resources/test.csv', help='train dataset')
    parser.add_argument('--model', '-m', default='model.pcl', help='file to save model into')
    parser.add_argument('--submission-sample', default='resources/sampleSubmission.csv', help='sample submission to fill in')
    parser.add_argument('--silent', '-s', default=False, help='silent mode', action='store_true')
    parser.add_argument('--file', '-f', default='submission.csv', help='file to write predictions into')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset, index_col=0)
    sub_df = pd.DataFrame(index=df.index, columns=['start.' + str(_) for _ in range(1, 401)])

    delta = df.iloc[:, 0]
    X = df.iloc[:, 1:401].values.reshape(-1, 20, 20)

    with open(args.model, 'rb') as f:
        model_map = pcl.load(f)

    kernel_size = 3
    for i in range(1, 6):
        if not args.silent:
            print(f"Predicting for delta={i}")
        X_i = X[delta == i]

        X_test = df_to_windows(X_i, kernel_size=kernel_size)
        X_test = X_test.astype(np.float64)

        rf_dict = model_map[i]
        rf = RandomForestClassifier()
        rf.trees = rf_dict['trees']
        rf.feat_ids_by_tree = rf_dict['feat_ids_by_tree']
        rf.unique_labels = rf_dict['unique_labels']

        test_predictions = rf.predict(X_test)
        sub_df[delta == i] = test_predictions.reshape(-1, 400)

    sub_df.to_csv(args.file)
    if not args.silent:
        print(f"Predictions are saved into {args.file}")