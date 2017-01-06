



### Installation Guide

Install Ubuntu 16.04 (for example from specially created USB stick) without auto updates

Update kernel after install: 
> sudo apt-get dist-upgrade

Install cuda which included the display driver (http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning)

The guide is for an amazon server, so not everything applies. Things I did different from the manual:

* Download and install cuda 8.0

* After extracting the package, when you install the display driver, you need to exit the window environment first. Do the following: 
Ctrl + Alt + F1

> sudo bash

> service ligthdm stop

> ./Downloads/NVIDIA-Linux-x86_64… 

> reboot

Install and update anaconda for python 2.7 (https://www.continuum.io/downloads)

Update all anaconda packages

> Conda install keras

Configure Keras to use Theano and set some Theano settings Follow step 3 and 4 from here: http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning To configure ~/.theanorc and ~/.keras/keras.json

> Conda install pyqt4 (downgrade) 

Download and install SharedArray python package: https://pypi.python.org/pypi/SharedArray

Compile and install OpenCV 2.4.13 (ensure GPU and ffmpeg library): First install all packages (compiler, required and optional) listed here: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation 
Then follow this user manual (without setting up virtual environments: http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support

Make sure there is a symbolic link in …/anaconda2/lib/python2.7/site_packages to the compiled opencv (cv2.so, probably in /usr/local/lib/python2.7/site-packages/cv2.so)

You can do a quick test for the installation from the terminal:
> python 

> import keras 

> import cv2

Whatever API you prefer, make sure it uses the anaconda python. For pycharm it as necessary to start it from the terminal, otherwise it would not use the path setting defined in ~/.bashrc
