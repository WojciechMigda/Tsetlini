# -*- coding: utf-8 -*-

# based on https://github.com/cair/pyTsetlinMachine/blob/dfaa7f36a5fa5cc852645277605358ae4d955898/examples/BreastCancerDemo.py
# and https://github.com/cair/pyTsetlinMachine/blob/dfaa7f36a5fa5cc852645277605358ae4d955898/pyTsetlinMachine/tools.py

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


class Binarizer:
    def __init__(self, max_bits_per_feature = 25):
        self.max_bits_per_feature = max_bits_per_feature
        return

    def fit(self, X):
        self.number_of_features = 0
        self.unique_values = []
        for i in range(X.shape[1]):
            uv = np.unique(X[:,i])[1:]
            if uv.size > self.max_bits_per_feature:
                unique_values = np.empty(0)

                step_size = 1.0*uv.size/self.max_bits_per_feature
                pos = 0.0
                while int(pos) < uv.size and unique_values.size < self.max_bits_per_feature:
                    unique_values = np.append(unique_values, np.array(uv[int(pos)]))
                    pos += step_size
            else:
                unique_values = uv

            self.unique_values.append(unique_values)
            self.number_of_features += self.unique_values[-1].size
        return

    def transform(self, X):
        X_transformed = np.zeros((X.shape[0], self.number_of_features))

        pos = 0
        for i in range(X.shape[1]):
            for j in range(self.unique_values[i].size):
                X_transformed[:,pos] = (X[:,i] >= self.unique_values[i][j])
                pos += 1

        return X_transformed


breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

b = Binarizer(max_bits_per_feature = 10)
b.fit(X)
X_transformed = b.transform(X)

output_data = np.c_[X_transformed, Y]
np.savetxt("BreastCancerData.txt", output_data, fmt="%d")
