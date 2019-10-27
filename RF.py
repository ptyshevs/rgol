import numpy as np
import pandas as pd
import multiprocessing as mp
import scipy as sp

import pyximport
pyximport.install(language_level=3)
from FastDecisionTree import *

def job(model, X, y, seed, kwargs):
    max_features = kwargs['max_features']
    verbose = kwargs['verbose']
    random_state = kwargs['random_state']

    sp.random.seed(seed=seed)
    n, m = X.shape
    nrange = np.arange(n)
    mrange = np.arange(m)
    fi = np.random.choice(mrange, replace=False, size=max_features)
    si = np.random.choice(nrange, replace=True, size=n)
    X_train = X[si, :][:, fi]
    y_train = y[si]
    if verbose:
        print(f"Fitting {seed - random_state}-th tree")
    
    model.fit(X_train, y_train)
    return model, fi

class RandomForestClassifier:
    def __init__(self, n_estimators=12, max_depth=5, max_features=None, 
                 random_state=42, verbose=False, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = mp.cpu_count()
        if n_jobs > 0:
            self.n_jobs = min(mp.cpu_count(), n_jobs)
        self.pool = mp.Pool(self.n_jobs)

        self.unique_labels = None
        self.trees = []
        self.feat_ids_by_tree = []
        
    def fit(self, X, y):
        if self.n_jobs > 1:
            return self.fit_parallel(X, y)
        self.trees = []
        self.feat_ids_by_tree = []
        n, m = X.shape
        nrange = np.arange(n)
        mrange = np.arange(m)
        max_features = m if self.max_features is None else self.max_features
        
        self.unique_labels = np.unique(y)
        for i in range(self.n_estimators):
            seed = self.random_state + i
            np.random.seed(seed)

            if self.verbose:
                print(f'[{i+1}/{self.n_estimators}]')

            fi = np.random.choice(mrange, replace=False, size=max_features)
            si = np.random.choice(nrange, replace=True, size=n)
            X_train = X[si, :][:, fi]
            y_train = y[si]

            self.feat_ids_by_tree.append(fi)
            
            dt = FastDecisionTree(max_depth=self.max_depth, random_state=seed)
            self.trees.append(dt)
            dt.fit(X_train, y_train)

        if self.verbose:
            print(f"Training is finished: {len(self.trees)} trees have beed built | {len(self.unique_labels)} labels in target")
        return self
    
    def fit_parallel(self, X, y):
        self.trees = []
        self.feat_ids_by_tree = []
        n, m = X.shape
        max_features = m if self.max_features is None else self.max_features
        self.unique_labels = np.unique(y)
        
        kwargs = {'verbose': self.verbose, 'max_features': max_features, 'random_state': self.random_state}
        tasks = [(FastDecisionTree(max_depth=self.max_depth, random_state=seed), X, y, seed, kwargs) for seed in range(self.random_state, self.random_state + self.n_estimators)]
        results = self.pool.starmap(job, tasks)
        for tree, fi in results:
            self.trees.append(tree)
            self.feat_ids_by_tree.append(fi)
        if self.verbose:
            print(f"Training is finished: {len(self.trees)} trees have beed built | {len(self.unique_labels)} labels in target")
        return self
    
    def predict_proba(self, X):
        probs = np.zeros((len(X), len(self.unique_labels)))
        for tree, fi in zip(self.trees, self.feat_ids_by_tree):
            X_sub = X[:, fi]
            probs += tree.predict_proba(X_sub)
        return probs / len(self.trees)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.get_params()}'

    def get_params(self):
        return {'n_estimators': self.n_estimators, 'max_depth': self.max_depth, 'max_features': self.max_features}
