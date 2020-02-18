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
import operator

# Load data
dataset = 'pubmed'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)

original_adj = adj
np_adj = adj.toarray()


# Removal based on edge betweenness
#G = nx.from_numpy_matrix(np_adj)
#centrality = nx.edge_betweenness(G)
#sorted_x = sorted(centrality.items(), key=operator.itemgetter(1)) # sort
#sorted_x.reverse() # reverse
#input = random.sample(list(range(1000)),int(np.ceil(3327 * 1/100))) # with replacement -> np.random.choice


sorted_x = pd.read_csv("ordered_edgelist_betweenness_pubmed.csv", index_col=0)
sorted_x = sorted_x.values[:,0:2]
sorted_x = sorted_x.astype(int)

selection_input = int(np.ceil(19717 * 2/100))
random_selection = random.sample(list(range(selection_input)), int(np.ceil(19717 * 1/100)))
for i in random_selection:
    np_adj[sorted_x[i,0]-1, sorted_x[i,1]-1] = 0
    np_adj[sorted_x[i,1]-1, sorted_x[i,0]-1] = 0

adj = csr_matrix(np_adj)


# Parameters
num_comp = 5
num_filter = 1
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0.
l2_reg = 5e-4
learning_rate = 1e-2
epochs = 2000
es_patience = 300
recur_num = 3 

# Normalized Laplacian 
#fltr = normalized_adjacency(adj, symmetric=True)

# Generalized form
sigma = 0.5
degrees = np.float_power(np.array(adj.sum(1)), -sigma).flatten()
degrees[np.isinf(degrees)] = 0.
normalized_D = sp.diags(degrees, 0)

degrees_sec = np.float_power(np.array(adj.sum(1)), (sigma - 1)).flatten()
degrees_sec[np.isinf(degrees_sec)] = 0.
normalized_D_sec = sp.diags(degrees_sec, 0)
fltr = normalized_D.dot(adj)
fltr = fltr.dot(normalized_D_sec)

# For validation and test
sigma = 0.5
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

dropout_1 = Dropout(dropout_rate)(X_in)
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
eval_results = model.evaluate([node_features, ori_fltr], # 07/06 YC DU library
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))



