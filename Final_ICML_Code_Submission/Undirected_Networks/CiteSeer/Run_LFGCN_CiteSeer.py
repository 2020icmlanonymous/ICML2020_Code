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
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import inspect
import random
import networkx as nx
from sys import getsizeof
import math
import operator


# Load data
dataset = 'citeseer'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

original_adj = adj
np_adj = adj.toarray()

# Removal based on edge betweenness 
G = nx.from_numpy_matrix(np_adj)
centrality = nx.edge_betweenness(G)
sorted_x = sorted(centrality.items(), key=operator.itemgetter(1)) # sort
sorted_x.reverse() # reverse
weights = []
for i in range(len(sorted_x)):
    weights.append(sorted_x[i][1])
scaling_weights = weights/np.sum(weights)


selection_input = int(np.ceil(3327 * 5/100))

for i in range(selection_input):
    np_adj[sorted_x[i][0][0], sorted_x[i][0][1]] = 0
    np_adj[sorted_x[i][0][1], sorted_x[i][0][0]] = 0

adj = csr_matrix(np_adj)

# Parameters
num_comp = 1
num_filter = 1
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0.75
l2_reg = 5e-4
learning_rate = 1e-2
epochs = 2000
es_patience = 150
recur_num = 3 

# Normalized Laplacian
#fltr = normalized_adjacency(adj, symmetric=True)

# Generalized form 
sigma = 0.001
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
gamma = 3.
degrees = np.array(adj.sum(1)).flatten()
degrees[np.isinf(degrees)] = 0.
D = sp.diags(degrees, 0)
L = D - adj
L_darray = L.toarray()
D, V = np.linalg.eigh(L_darray, 'U')
M_gamma_Lambda = D
M_gamma_Lambda[M_gamma_Lambda < 1e-5] = 0
M_V = V
M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
M_V = sp.csr_matrix(M_V)
Lg = M_V * M_gamma_Lambda
Lg = Lg * sp.csr_matrix.transpose(M_V)
Lg = Lg.toarray()
Lg = Lg.reshape(1, -1)
Lg[abs(Lg) < 1e-5] = 0.
Lg = Lg.reshape(N, -1)
Dg = np.diag(np.diag(Lg))
Ag = Dg - Lg
Ag = sp.csr_matrix(Ag)
alpha = 0.5
power_Dg = np.float_power(np.diag(Dg), -alpha)
power_Dg = np.diag(power_Dg)
power_Dg = sp.csr_matrix(power_Dg)
power_Dg_right = np.float_power(np.diag(Dg), (alpha - 1))
power_Dg_right = np.diag(power_Dg_right)
power_Dg_right = sp.csr_matrix(power_Dg_right)
fltr = power_Dg * Ag
fltr = fltr * power_Dg_right
"""

# For validation and test
sigma = 0.55
ori_degrees = np.float_power(np.array(original_adj.sum(1)), -sigma).flatten()
ori_degrees[np.isinf(ori_degrees)] = 0.
ori_normalized_D = sp.diags(ori_degrees, 0)

ori_degrees_sec = np.float_power(np.array(original_adj.sum(1)), (sigma - 1)).flatten()
ori_degrees_sec[np.isinf(ori_degrees_sec)] = 0.
ori_normalized_D_sec = sp.diags(ori_degrees_sec, 0)
ori_fltr = ori_normalized_D.dot(original_adj)
ori_fltr = ori_fltr.dot(ori_normalized_D_sec)


# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(0.75)(X_in)
graph_conv_1 = FGSConv(32,
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
                       num_comp=2,
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
validation_data = ([node_features, ori_fltr], y_val, val_mask)


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
eval_results = model.evaluate([node_features, ori_fltr],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))


