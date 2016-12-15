
## Deep Learning FAQ


* What is deep learning? Deep learning is a branch of machine learning algorithms that currently delivering state-of-the-art results in most areas inclduing vision, speech, text,etc.
* What is difference between deep learning and machine learning? Deep learning is a branch of machine learning.
* What is convolutioanl neural networks? Also called CNN, is the main building block in current deep learning architectures.
* What is CNN? Also called Convolutional Neural Networks.
* What is supervised leaening? It is currently the main machine learning paradigm. When you have both data and labels you can do supervised learning to learn a function that maps your data to the corresponding labels. 
* What is unsupervised learning? It is another paradigm in machine learning. When you have data but do not have labels you can do unsupervided learning to find patterns in your data.
* What is reinforcement learning? It is another paradigm of machine learning, mainly used for robotics, agents playing games, etc. When you have data (for ex. screen) and the final goal (for ex. win the game) but do not have labels you can do RL. 
* What is RL? Also called reinforcement learning.
* What is generative adversarial networks? Also called GAN, a new type of deep learning architectures for generating real-looking data. They can be used in different ways including creating extra data to be used for training, segmentation, etc. Yan LeCun thinks GAN is the most interesting idea in the last 10 years in ML.  
* What is GAN? Also called generative adversarial networks.
* What is Caffe, Tensorflow, Theano, Torch, etc? All are software platforms for developing deep learning networks. IF you want to hard code your deep learning networks you do not need them but it is a waste of time if you do that.
* Which deep learning framework I should use? Use Keras, it is simple, flexible and to the point.
* Where should I start learning deep learning? Start from this book: deep learning with Python, Jason Brownlee.
* What is AGI or GAI? Artificial General Intelligence, General Artificial Intelligence
* what is training? Developing machine learning algorithms using collected data and labels by optimizing a cost/loss function.
* Why do we train deep learning networks? Without training the network will only predict random outputs. So, to predict meaningful outputs, we need to train the networks.
* Why do we collect data? Depending on application, Internet, hospitals, manual data collection, etc.
* Why should we label or annotate the data? The raw data without labels cannot be used for optimizing/ training in supervised learning unless it is labeled.
* What is cross-validation? Validating our algorithm on some test data that has not been used for training.
* What is ensembling? Combining multiple ML algorithms to improve accuracy/performance.
* What is hyper-parameters optimization? Parameters such as learning rate, batch size, network size, etc., need to be fine-tuned for the best accuracy. This is called hyper-parameter optimization.
* What computer should I use for training? A powerful computer with Nvidia GPU and enough RAM (>= 8GB).
* What do I do after training? Deploy your DL network for the desired task.
* What is learning rate? An important hyper-parameter that is used during training for optimizing the loss/cost function. Typical value is 3E-4 for Adam optimizer and 1E-3 for SGD.
* How should I set the learning rate? Start from a common value such as 1E-3 or 1E-4 and gradually tweak if needed.
* What is batch size? It is the size of data that is used in each iteration of optimization/training. Typical value is 32.
* What should I use for batch size? Start from small values such as 8,16,32.
* What is SGD, ADAM, etc? Stochastic Gradient Descent and their variations. These are different optimizers cane be used during training. 
* Which optimizer method should I use? Start with Adam or SGD.
* What is fine-tunning? Adjusting/optimizing the training weights to achieve a better performance. In fine-tuning you do not start from random initialization but from some pre-trained values.
* Who is Yan LeCun? He is basically the father of conv nets, one of the AI pioneers.
* Who is Andrew Ng? He is stanford professor, coursera co-founder and one of the AI pioneers.
* Who is Jeff Hinton? Hinton is long-term AI pioneer. He ignited the field in 2006.
* Who is Bengio? One of the AI pioneer based on U of Montreal.
* Who are the celebrities in deep learning? The aboves.
* Is deep learning equivalent to AI? It is being used interchangebly these days.
* What is pre-training? It means training your DL network with some data not necessarily the actual data from the application. 
* What language should I use? Python.
* What is Jupyter notebook? Cool IDE for code development.
* What is numpy? Numerical Python, one of the python libraries for numerical calculations.
* What is matplotlib? Cool Python package for plotting.
* What is opencv? Another C++/Python package for dealing with images and videos.
* What is overfitting? When your algorithms works perfectly on training data and fails on test data.
* How can I avoid overfitting? Set apart a validation/test data, monitor training and test error during training, stop the training if no improvement is seen in the test error, use data augmentation, use drop-out.
* What is regularization? A technique for avoiding over-fitting. 
* What is data augmentation? Increasing the size of training data with various techniques.
* What is drop out? A technique to avoid overfitting. Turining off some of the neurons randomly during training.
* What is loss function or objective function? It is the function being optimized during training.
* what is backpropagation? It is an algorithm used for training of DL/ML networks. It is based on calculating the error between the ground truth and network's output and propagating the error to adjust the weights in the network.
* what is gradient descent? It is an optimization algorithm. It's variant stochastic gradient descent is used for training DL networks.
* what is gradient checking? Making sure that you calculated gradients correctly. Back then, when there was no platform such as caffe, tensorflow,etc., you had to hardcode the backpropagation alogirithm, gradients in each layer, etc. To make sure that you have done these correctly, you better did gradient checking. Although, there is a saying that "backpropagation is such a robust algorithm that any bug you made in development would act like a regulizer!!"
* what is AlexNet? AlexNet is a CNN based classification network that outperformed the benchmarks in ILSVRC 2012 competition by a large margin. AlexNet was laid the foundation for many image recognition breakthroughs. 
* What VGG net?It is another CNN based classfication network won 2014 competitions.
* what is ResNet? Same a above won 2015 competition. 
* What is RNN? Recurrent neural networks, used for various applications dealing with time-series data.
* What is LSTM and GRU? Large short-term memory and gated recurrent unit, two types of RNN nets.
* What is Q-learning? It is reinforcement learning algorithm developed by Google Deepmind.
* What is dilated convolution? It is a type of CNN to increase the receptive field without increasing the number of parameters.
* what is max pooling and average pooling? Max/averag pooling is the layer we use after every convolution layer to reduce the spatial resolution of features. Max pooling is more common to use these days.
* What is ReLU, Leaky ReLU? Rectified linear unit and its leaky version. These are activation functions used in DL networks. 
* what is sigmoid? Sigmoid the s-shape activation function commonly used in ML.
* What is activation function?






















