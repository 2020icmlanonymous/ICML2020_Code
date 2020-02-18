import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from spektral_fix.datasets import citation
from spektral_fix.layers import FGSConv
from spektral_fix.utils import normalized_laplacian, rescale_laplacian, normalized_adjacency, degree_power
from sklearn.model_selection import GridSearchCV
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import pandas as pd
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
import warnings
import itertools
from utils import load_dataset, score_link_prediction, score_node_classification
import pickle as pkl
import networkx as nx
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import inspect


# Load data - Directed CORA #
g = load_dataset('cora.npz')
A, X, z = g['A'], g['X'], g['z']
label_mat = np.zeros((19793, 70),dtype= float)

idx = np.arange(len(z))
idx_train = pd.read_csv("idx_train.csv")
idx_train = idx_train.values
idx_train = idx_train.reshape((1979,))

idx_val = pd.read_csv("idx_val.csv")
idx_val = idx_val.values
idx_val = idx_val.reshape((1979,))

idx_test = pd.read_csv("idx_test.csv")
idx_test = idx_test.values
idx_test = idx_test.reshape((1979,))
print(idx_train)
print(idx_test)

y_train = np.zeros((19793, 70),dtype= float)
for i in idx_train:
    y_train[i,z[i]] = 1.

y_val = np.zeros((19793, 70), dtype= float)
for i in idx_val:
    y_val[i, z[i]] = 1.

y_test = np.zeros((19793, 70), dtype= float)
for i in idx_test:
    y_test[i, z[i]] = 1.

train_mask = np.full((z.shape[0],), False, dtype= bool)
val_mask = np.full((z.shape[0],), False, dtype= bool)
test_mask = np.full((z.shape[0],), False, dtype= bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True
A_darray = A.toarray()
transposed_A_darray = np.transpose(A_darray)
sym_A_darray = A_darray + np.multiply(transposed_A_darray,(transposed_A_darray > A_darray)) - np.multiply(A_darray, (transposed_A_darray > A_darray))
A = csr_matrix(sym_A_darray)

adj = A; node_features = X


# Parameters
num_comp = 1
num_filter = 1
recurrent = None
N = node_features.shape[0]

n_classes = y_train.shape[1]
dropout_rate = 0.
l2_reg = 5e-4
learning_rate = 1e-1
epochs = 1000
es_patience = 200
recur_num = 3

#fltr = normalized_adjacency(adj, symmetric=True)

# G-SSL part
sigma = 1.0 
degrees = np.float_power(np.array(adj.sum(1)), -sigma).flatten()
degrees[np.isinf(degrees)] = 0.
normalized_D = sp.diags(degrees, 0)

degrees_sec = np.float_power(np.array(adj.sum(1)), (sigma - 1)).flatten()
degrees_sec[np.isinf(degrees_sec)] = 0.
normalized_D_sec = sp.diags(degrees_sec, 0)
fltr = normalized_D.dot(adj)
fltr = fltr.dot(normalized_D_sec)


# Fractional part
"""
gamma = 0.001
degrees = np.array(adj.sum(1)).flatten()
degrees[np.isinf(degrees)] = 0.
D = sp.diags(degrees, 0)
L = D - adj


L_darray = L.toarray()
D, V = np.linalg.eigh(L_darray, 'U')
M_gamma_Lambda = D
M_gamma_Lambda[M_gamma_Lambda < 1e-10] = 0
M_V = V

M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
M_V = sp.csr_matrix(M_V)
Lg = M_V * M_gamma_Lambda
Lg = Lg * sp.csr_matrix.transpose(M_V)

Lg = Lg.toarray()
Lg = Lg.reshape(1, -1)
Lg[abs(Lg) < 1e-10] = 0.
Lg = Lg.reshape(N, -1)
Dg = np.diag(np.diag(Lg))
Ag = Dg - Lg
Ag = sp.csr_matrix(Ag)

alpha = 0.9
power_Dg = np.float_power(np.diag(Dg), -alpha)
power_Dg = np.diag(power_Dg)
power_Dg = sp.csr_matrix(power_Dg)

power_Dg_right = np.float_power(np.diag(Dg), (alpha - 1))
power_Dg_right = np.diag(power_Dg_right)
power_Dg_right = sp.csr_matrix(power_Dg_right)

fltr = power_Dg * Ag
fltr = fltr * power_Dg_right
"""

# PCA
node_features_darray = node_features.toarray()
n_components = 130
F = n_components
pca = PCA(n_components= n_components)
node_features_PCA = pca.fit_transform(node_features_darray)
node_features = csr_matrix(node_features_PCA)


# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = FGSConv(128,
                        num_comp=num_comp,
                        num_filter=num_filter,
                        recurrent=recurrent,
                        recur_num = recur_num,
                        dropout_rate=dropout_rate,
                        activation='elu',
                        gcn_activation='elu',
                        kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])


dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = FGSConv(n_classes,
                        num_comp=1,
                        num_filter=1,
                        recurrent=recurrent,
                        recur_num = recur_num,
                        dropout_rate=dropout_rate,
                        activation='softmax',
                        gcn_activation=None,
                        kernel_regularizer=l2(l2_reg))([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()


callbacks = [
    EarlyStopping(monitor='val_weighted_acc', patience=es_patience),
    ModelCheckpoint('best_model.h5', monitor='val_weighted_acc',
                    save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = ([node_features, fltr], y_val, val_mask)


model.fit([node_features, fltr],
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=callbacks)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features, fltr],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
