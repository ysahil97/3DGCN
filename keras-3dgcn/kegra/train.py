from __future__ import print_function

from keras.layers import Input, Dropout, Flatten, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import pdb
import numpy as np
from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import time

def get_dim():
    X, A, y = load_data(49)
    return X.shape[1], y.shape[1]

# Define parameters
DATASET = 'cora'
FILTER = 'chebyshev'  # 'chebyshev'
IS_DILATED =True
MAX_DEGREE = 4  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

if FILTER == 'localpool':
    support = 1
    G = [Input(shape=(None, None), batch_shape=(None, None))]
elif FILTER == 'chebyshev':
    support = MAX_DEGREE + 1
    G = [Input(shape=(None, None), batch_shape=(None, None)) for _ in range(support)]



sizes = get_dim()
X_in = Input(shape=(sizes[0],))
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(4, True, support,  activation='relu', kernel_regularizer=l2(5e-4),use_bias=False)([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(sizes[1], False, support, activation='softmax',use_bias=False)([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

t_loss = 0
t_acc = 0
for iddx in range(4,114,5):
    # Get data
    X, A, y = load_data(iddx)
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    # Normalize X
    # Not Needed for 3dgcn as in our dataset, we are assuming features to be identity
    #X /= X.sum(1).reshape(-1, 1)

    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        if IS_DILATED:
            A_ = preprocess_adj_dilated(A, SYM_NORM)
        else:
            A_ = preprocess_adj(A, SYM_NORM)
        num_batches = len(A_)
        Aa = (A_)
        Aaa = Aa.shape[0]
        graph = [X, Aa]

    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        if IS_DILATED:
            T_k = chebyshev_polynomial_dilated(L_scaled, MAX_DEGREE)
        else:
            T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        Aaa = (T_k[0]).shape[0]
        graph = [X]+T_k

    else:
        raise Exception('Invalid filter type.')

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999
    # Fit
    for epoch in range(1, NB_EPOCH+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, np.array(y_train), sample_weight=train_mask,
                  batch_size=Aaa, epochs=1, shuffle=False, verbose=0)
        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    t_loss += test_loss[0]
    t_acc += test_acc[0]
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))

print("Final Test set results:",
      "loss= {:.4f}".format(t_loss/20),
      "accuracy= {:.4f}".format(t_acc/20))
