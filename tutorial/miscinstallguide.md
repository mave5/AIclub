
## Misc Installations Guide



#### Keras
* get version: python -c "import keras; print keras.__version__"
* install: sudo pip install keras
* upgrade: sudo pip install --upgrade keras
* [Dynamically switch Keras backend in Jupyter notebooks](http://www.nodalpoint.com/switch-keras-backend/)
* [Keras - Dockerfile](https://gist.github.com/fiskio/638c2ded94bef1be39b4)

#### Theano
* [Theano Configuration](http://deeplearning.net/software/theano/library/config.html)
* [Theano config directly in script](http://stackoverflow.com/questions/33988334/theano-config-directly-in-script)

#### python/anaconda
* matplotlib: conda install -c conda-forge matplotlib=2.0.0 
* opencv: conda install -c https://conda.binstar.org/menpo opencv
* skimage: conda install scikit-image
* MKL: conda install nomkl numpy scipy scikit-learn numexpr 

       conda remove mkl mkl-service
* imageio: conda install -c conda-forge imageio
* imageio: pip install imageio
* sk-video:  pip install sk-video



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
* [change hostname in terminal](https://www.cyberciti.biz/faq/ubuntu-change-hostname-command/)
* Change volume/disk label, [reference](https://ubuntuforums.org/showthread.php?t=1113236)

       - get device info: sudo fdisk -l
       - change label: sudo e2label /dev/sdb1 "new_label"

* memory,cpu usage GUI: system monitor
* memory,cpu usage terminal: top
* nvidia GPU usage: sudo watch nvidia-smi
* mount a network drive:
       - sudo mkdir path_to_your_mount_name
       - add the following line to /etc/fstab file to auto mount the drive
       - //10.138.53.27/db /medica/your_mount_name cifs username=read_db,password=your_password,uid=1000,gid=1000
       - sudo mount -a
* mount a drive:
       - open /etc/fstab and add next line to the end
       - Example: /dev/sdc1  /media/mount_name  ext4 defaults 0 0

* change owner 
       - sudo chown usrnm:root -R dirName

#### GitHub
* get repo history: git log --oneline
* reset to a previous state: git reset --hard commit_head
* go to previous state: git checkout commit_head
* go to current state: git checkout master
* [Github Undoing Changes](https://www.atlassian.com/git/tutorials/undoing-changes/git-reset)
* store credentials

       $ git config credential.helper store
       $ git push repo_url

       Username for 'https://github.com': <USERNAME>
       Password for 'https://USERNAME@github.com': <PASSWORD>



#### Java
* [Install Java 8](http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/#)


