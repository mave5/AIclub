### Install Keras for Windows

* [install anaconda] (https://www.continuum.io/downloads)
* install other libraries:  conda install mingw libpython
* install spyder: conda install spyder
* install OpenCV:
* Copy cv2.pyd to C:\Anaconda2\Lib\site-packages
* test import numpy, scipy and cv2
* [install git] (https://git-for-windows.github.io/)
* install theano: conda install theano
* test Theano: run Theano_Test.py
* Config theano:
 add the following to file: .theanorc.txt: 
   ..* [global]
   ..* floatX = float32
   ..* device = cpu
 
* install keras
               - cd C:\Anaconda2\Lib\site-packages
               - git clone git://github.com/fchollet/keras.git
               - cd keras
               - python setup.py develop
 
               or
               - go ti github (address above) and download keras
               - create a folder called keras in C:\Anaconda2\Lib\site-packages
               - past all in the setup.py in that folder form where it was downloaded
               - cd C:\Anaconda2\Lib\site-packages\keras
               - python setup.py develop
 
* test keras
               - run Keras_Test.py
 
