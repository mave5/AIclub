# docker environment

FROM quay.io/domino/base:2016-12-07_1239

USER root

RUN \
    export CUDA_HOME=/usr/local/cuda-7.5 && \
    export CUDA_ROOT="/usr/local/cuda-7.5/bin" && \
    export PATH=/usr/local/cuda-7.5/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64 && \
    cd ~ && \
    wget https://app.dominodatalab.com/johnjoo/cuDNN_for_TF/raw/latest/cudnn-7.5-linux-x64-v5.1.tgz -O cudnn-7.5-linux-x64-v5.1.tgz.gz && \
    gunzip cudnn-7.5-linux-x64-v5.1.tgz.gz && \
    tar xvzf cudnn-7.5-linux-x64-v5.1.tgz && \
    cp cuda/include/cudnn.h /usr/local/cuda-7.5/include && \
    cp cuda/lib64/libcudnn* /usr/local/cuda-7.5/lib64 && \
    chmod a+x /usr/local/cuda-7.5/include/cudnn.h /usr/local/cuda-7.5/lib64/libcudnn* && \
    rm -rf cuda && \

    pip install 'certifi==2015.4.28' --force-reinstall && \
  export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl && \
    pip install --ignore-installed  --upgrade $TF_BINARY_URL && \
    pip install keras

RUN sudo apt-get update 
RUN apt-get install -y libhdf5-dev libyaml-dev pkg-config  libopencv-dev python-opencv 

RUN \
export CUDA_ROOT="/usr/local/cuda-7.5/bin"  && \
export PATH=/usr/local/cuda-7.5/bin:$PATH && \
export LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/lib:/usr/local/lib:" && \

pip install --upgrade six && \
cd ~ && \
git clone https://github.com/NervanaSystems/neon.git && \
cd neon && \
 # export PATH=/usr/local/cuda-7.5/bin:$PATH && \
  make sysinstall HAS_GPU=true

RUN pip install Theano
RUN pip install opencv-python

# Update R
RUN apt-get update && apt-get install -y r-base r-base-dev
RUN R -e 'install.packages(c("data.table","ggfortify","gridExtra","sitools","shinydashboard","shinyjs"))'
