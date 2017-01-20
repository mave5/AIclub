
## Misc Installations Guide



#### Keras
* get version: python -c "import keras; print keras.__version__"
* install: sudo pip install keras
* upgrade: sudo pip install --upgrade keras



#### CUDA
* check cuda version: nvcc --version
* check cudnn version:  cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2


#### IP address 
* get ip address ubuntu: ip route get 8.8.8.8 | awk '{print $NF; exit}'
* get ip address ubuntu: hostname -I



#### Linux
* kernel version: uname -r


#### GitHub
* get repo history: git log --oneline
* reset to a previous state: git reset --hard commit_head
* go to previous state: git checkout commit_head
* go to current state: git checkout master
* [Github Undoing Changes] (https://www.atlassian.com/git/tutorials/undoing-changes/git-reset)