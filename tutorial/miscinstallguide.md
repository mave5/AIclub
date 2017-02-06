
## Misc Installations Guide



#### Keras
* get version: python -c "import keras; print keras.__version__"
* install: sudo pip install keras
* upgrade: sudo pip install --upgrade keras
* [Dynamically switch Keras backend in Jupyter notebooks] (http://www.nodalpoint.com/switch-keras-backend/)
* [Keras - Dockerfile] (https://gist.github.com/fiskio/638c2ded94bef1be39b4)


#### CUDA
* check cuda version: nvcc --version
* check cudnn version:  cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
* remove cuda toolkit: sudo apt-get remove nvidia-cuda-toolkit
* remove cuda dependencies: sudo apt-get remove --auto-remove nvidia-cuda-toolkit


#### IP address 
* get ip address ubuntu: ip route get 8.8.8.8 | awk '{print $NF; exit}'
* get ip address ubuntu: hostname -I



#### Linux
* kernel version: uname -r
* [How to create a bootable USB stick on Ubuntu](https://www.ubuntu.com/download/desktop/create-a-usb-stick-on-ubuntu)
* number of certain file type in a dir: ls -lR /path/to/dir/*.jpg | wc -l
* delete dir: rm -R dir_name


#### GitHub
* get repo history: git log --oneline
* reset to a previous state: git reset --hard commit_head
* go to previous state: git checkout commit_head
* go to current state: git checkout master
* [Github Undoing Changes] (https://www.atlassian.com/git/tutorials/undoing-changes/git-reset)



#### Java
* [Install Java 8] (http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/#)


