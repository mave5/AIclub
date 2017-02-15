


## Access to Keras model and graph in Tensorflow


import keras.backend as K

import tensorflow as tf

# access to session

sess=K.get_session()

# grab graph

graph=K.get_session().graph
