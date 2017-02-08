## Install Keras with Tensorflow backend and GPU support


#### System
 Ubuntu 16.04 LTS, 4X GPU: Titan X

[Source: Tensorflow webpage](https://www.tensorflow.org/get_started/os_setup#anaconda_installation)


 * [Install anaconda] (https://www.continuum.io/downloads)
 
 * Create a conda environment called tensorflow:
    $ conda create -n tensorflow python=2.7

* Activate environment
  $ source activate tensorflow
  
  (tensorflow)$  # Your prompt should change

*   Install CUDA toolkit 8.0 and CuDNN v5. 

* Intstall tensorflow:

 (tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-  linux_x86_64.whl

 (tensorflow)$ pip install --ignore-installed --upgrade $TF_BINARY_URL

* Install ipython

 (tensorflow)$ conda install ipython

* Install jupyer notebook

 (tensorflow)$ conda install jupyter

* Install keras

 (tensorflow)$ pip install keras
