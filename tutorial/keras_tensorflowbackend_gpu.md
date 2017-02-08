## Install Keras with Tensorflow backend and GPU support


#### System
 Ubuntu 16.04 LTS, 4X GPU: Titan X

[Source: Tensorflow webpage](https://www.tensorflow.org/get_started/os_setup#anaconda_installation)


 * Install latest Nvidia driver, cuda toolkit 8.x and cuDnn 5.x 
 
 * [Install anaconda] (https://www.continuum.io/downloads)
 
 * Create a conda environment called tensorflow
 
    $ conda create -n tensorflow python=2.7

* Activate environment

  $ source activate tensorflow
  
  (tensorflow)$  # Your prompt should change

*   Need CUDA toolkit 8.0 and CuDNN v5. 

* Intstall tensorflow:

 (tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-  linux_x86_64.whl

 (tensorflow)$ pip install --ignore-installed --upgrade $TF_BINARY_URL

* Install ipython

 (tensorflow)$ conda install ipython

* Install jupyer notebook

 (tensorflow)$ conda install jupyter

* Install keras

 (tensorflow)$ pip install keras
 
 
* Install other missing packages such as matplotlib,cv2,h5py,etc using conda
 
   matplotlib: (tensorflow)$ conda install -c conda-forge matplotlib=2.0.0
  
   opencv: (tensorflow)$ conda install -c https://conda.binstar.org/menpo opencv
  
   skimage: (tensorflow)$ conda install scikit-image

   MKL: (tensorflow)$ conda install nomkl numpy scipy scikit-learn numexpr

    (tensorflow)$ conda remove mkl mkl-service

 * Deactivate whenever done with environment
   source deactivate
   
 * Activate environment whenever needed:
   source activate tensorflow
   
 
