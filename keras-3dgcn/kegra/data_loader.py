from __future__ import print_function


from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.utils import *

SYM_NORM = True
DATASET = 'STWalk'

X, A, y = load_data(4)

#Localpool step
A_ = preprocess_adj(A, SYM_NORM)
support = 1
num_batches = len(A_)
graph = [X, A_]
G = [Input(shape=(None, None, None), batch_shape=(None, None, None), sparse=True) for _ in range(num_batches)]

print(X)
print(A_)
print(y)