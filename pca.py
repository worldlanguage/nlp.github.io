""" Import the basic requirements package """

import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

""" Dataset export function """

def read_csv():
    X, y = load_digits(return_X_y=True)

    print(X.shape)  # Print feature set and label set length

    return X

""" Dataset Normalization function """

def normalization(X):
    columns = 'col_' + pd.Series(np.arange(1, 1 + X.shape[1]).astype(str)).values
    data = pd.DataFrame(X, columns=columns)
    data = pd.DataFrame(data.apply(lambda x_row: (x_row - np.mean(x_row)) / math.sqrt(np.var(x_row)), axis=1).values, columns=columns)

    return data.values

""" Dataset Covariance function """

def covariance(X):

    X_cov = np.cov(np.transpose(X))

    return X_cov

""" Dataset Eig function """

def eig_values(X):

    a, b = np.linalg.eig(X)

    X_eig = pd.DataFrame()

    X_eig['Lambda'] = a
    X_eig['Beta_i'] = np.arange(X.shape[0])
    X_eig['Beta'] = X_eig['Beta_i'].apply(lambda i: b[i])

    X_eig.sort_values(by='Lambda', ascending=False)

    return X_eig


""" PCA model function """


# create model
def PCAVectorizer(decomposition=10):
    pca_params = {
        'decomposition': decomposition,  # Set the base of the idf logarithm
    }
    return pca_params


# fit model
def fit_transform(pca_params, X):
    X = normalization(X)

    X_cov = covariance(X)

    X_eig = eig_values(X_cov)
    X_eig = np.vstack(X_eig.sort_values(by='Lambda', ascending=False)[ :pca_params['decomposition']]['Beta'].values)

    pca_X = np.transpose(np.dot(X_eig, np.transpose(X)))

    columns = 'col_' + pd.Series(np.arange(1, 1 + pca_params['decomposition']).astype(str)).values
    submit = pd.DataFrame(pca_X, columns=columns)
    print(submit.head(20))

    return pca_X

""" PCA model training host process """

if __name__ == '__main__':
    sta_time = time.time()

    X = read_csv()

    model = PCAVectorizer(decomposition=6)

    pca_X = fit_transform(model, X)

    print("Time:", time.time() - sta_time)
