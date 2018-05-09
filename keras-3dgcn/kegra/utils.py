from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import pdb
import pickle as pk
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(index):
    """Load citation network dataset (cora only for now)"""

    adj_matrix = []
    adj_matrices = pk.load(open("./data/EPINION/result_"+str(index)+".dat","rb"))
    for i in range(len(adj_matrices)):

        # build symmetric adjacency matrix
        adj_matrices[i] = adj_matrices[i] + adj_matrices[i].T.dot(adj_matrices[i].T > adj_matrices[i]) - adj_matrices[i].dot(adj_matrices[i].T > adj_matrices[i])

    labels = pk.load(open("./data/EPINION/labels_3dgcn/labels_"+str(index)+".dat","rb"))
    labels = encode_onehot(labels[0].values())

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = np.identity(adj_matrices.shape[1])
    for i in range(adj_matrices.shape[0]-1):
     features = np.concatenate((features,np.identity(adj_matrices.shape[1])),axis=1)
    return features, adj_matrices, labels

def normalize_adj(adj, symmetric=True):
    if symmetric:
        a_norm = []
        # pdb.set_trace()
        for i in range(len(adj)):
            d = np.diag(np.divide(np.ones(adj[i].shape[0]),np.maximum(np.array(adj[i].sum(1)),np.full(adj[i].shape[0],1e-8)).flatten()), 0)
            adj[i] = adj[i].dot(d).transpose().dot(d)
    else:
        a_norm = []
        for i in range(len(adj)):
            d = sp.diags(np.power(np.array(adj[i].sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adj[i]).tocsr()
    return adj


def preprocess_adj(adj, symmetric=True):
    for i in range(len(adj)):
        adj[i] = adj[i] + np.identity(adj[i].shape[0])
    adj = normalize_adj(adj, symmetric)
    x = adj[0]
    for i in range(len(adj)):
        if i != 0:
            x = np.concatenate((x,adj[i]),axis=1)
    return x


def preprocess_adj_dilated(adj, symmetric=True):
    for i in range(len(adj)):
        adj[i] = adj[i] + np.identity(adj[i].shape[0])
    adj = normalize_adj(adj, symmetric)
    adj = adj.T
    x = adj[0]
    for i in range(len(adj)):
        if i != 0:
            x = np.concatenate((x,adj[i]),axis=1)
    return x


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(17)
    idx_val = range(17, 22)
    idx_test = range(22, 25)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    for i in range(len(adj)):
        adj[i] = sp.eye(adj[i].shape[0]) - adj_normalized[i]
    return adj


def rescale_laplacian(laplacian):
    eigvals = []
    scaled_laplacian = laplacian.copy()
    for i in range(len(laplacian)):
        try:
            print('Calculating largest eigenvalue of normalized graph Laplacian...')
            largest_eigval = eigsh(laplacian[i], 1, which='LM', return_eigenvectors=False)[0]
        except ArpackNoConvergence:
            print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
            largest_eigval = 2
        scaled_laplacian[i] = (2. / largest_eigval) * laplacian[i] - sp.eye(laplacian[i].shape[0])
    return scaled_laplacian


def chebyshev_polynomial_dilated(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    T_k = list()

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = X.copy()
        return np.multiply(2.0, np.dot(X_,T_k_minus_one)) - T_k_minus_two

    for i in range(len(X)):
        Yu = list()
        Yu.append([np.identity(X[i].shape[0])])
        Yu.append([X[i]])


        for j in range(2, k+1):
            Yu.append([chebyshev_recurrence(Yu[-1][0], Yu[-2][0], X[i])])


        if i == 0:
            T_k = Yu.copy()
        else:
            T_kp = [m+n for m,n in zip(T_k,Yu)]
            T_k = T_kp.copy()
    T_k_final = []
    for i in range(len(T_k)):
        T_k[i] = np.array(T_k[i]).T
        x = T_k[i][0]
        for j in range(len(T_k[i])):
            if j != 0:
                x = np.concatenate((x,T_k[i][j]),axis=1)
        T_k_final.append(x)

    return T_k_final


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    T_k = list()

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = X.copy()
        return np.multiply(2.0, np.dot(X_,T_k_minus_one)) - T_k_minus_two
    for i in range(len(X)):
        Yu = list()
        Yu.append(np.identity(X[i].shape[0]))
        Yu.append(X[i])


        for j in range(2, k+1):
            Yu.append(chebyshev_recurrence(Yu[-1], Yu[-2], X[i]))
        Yu = np.array(Yu)

        if i == 0:
            T_k = Yu.copy()
        else:
            T_kp = []
            for j in range(len(Yu)):
                T_kp.append(np.concatenate((T_k[j],Yu[j]),axis=1))
            T_k = T_kp.copy()
    return T_k

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
