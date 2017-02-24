


## Access to Keras model in Tensorflow


import keras.backend as K

import tensorflow as tf


your keras model here (complited and loaded weights)


### access to session

sess=K.get_session()


### grab graph

graph=K.get_session().graph


### get graph definition
graph_def=graph.as_graph_def()


### export graph
meta_graph_def=tf.train.export_meta_graph(path2meta)


#### import graph
meta_graph_def=tf.train.export_meta_graph(path2meta)


### Using a single GPU on a multi-GPU system with Keras
    with tf.device('/gpu:2'):
        keras model definition here

    model.fit()  
