
# AIclub meeting summary

-----

### Meeting #10, 1/12/17

We discussed the following paper:
[Learning From Noisy Large-Scale Datasets With Minimal Supervision] (https://arxiv.org/abs/1701.01619)

The paper introduces a method for classification networks to take advantage of a large but noisy annotations to improve their accuracy.
In this work, a dataset with 9M images with noisy annotations and only 40K images with cleaned annotations are used. The idea is to train a network to learn the noise pattern and then use the network to clean the noisy annotations. Then both the clean data and the reduced-noise data are used for training. The authors claim that their approach is more effective than the traditional pre-training with noisy data and then fine-tunning with the clean data. Their results supports the claim. The slides can be seen here.


---------------

### Meeting #9, 1/6/17, JM

We discussed the following paper:
[Intriguing properties of neural networks] (https://arxiv.org/pdf/1312.6199v4)

The paper discusses 2 somewhat surprising properties of neural networks:
1. The idea that a particular neuron in the higher layers of the network stores specific semantic information is too simplistic. It turns out that semantic information is encoded in a many neurons together.
2. Using the gradients from a forward pass, it is possible to calculate the changes that need to be made to an image in order for it to be miss-classified with very high probability (adverserial example). These changes turn out to be surprisingly small (not visible) and they also seem to generalize to a worrying degree to across both different architectures and networks trained on different data. 

In the discussion we thought that GANs might help for the network to become more robust against adversarial examples. The authors seemed to have gone more into GAN research. Good topic for future meetings.

 --------------
 
### Meeting #8, 12/21/16, TY

We discussed the following three papers:
[A Comparison of Models for Predicting Early Hospital Readmission] (http://www.sciencedirect.com/science/article/pii/S1532046415000969)
[Dr. AI: Predicting Clinical Events via Recurrent Neural Network] (https://arxiv.org/pdf/1511.05942v11.pdf)
[Learning to Diagnose with LSTM Recurrent Neural Network] (https://arxiv.org/pdf/1511.03677v6.pdf) 

These three papers show a series of effort to use Machine Learning to predict patient outcome over existing public or private EMR data. These EMR data serve a good secondary use in Machine Learning to drive down healthcare cost. The first paper shows using a simple RBM consistently out-perform traditional statistica methods to predict the chance of hospital readmission. It does not explore time sequence medical data because RBM lacks the necessary construct.

The second paper is the first use or RNN with GRU over public patient data. It shows promising result of exploring the time sequence medical data in predicting the future diagnosis.

The third paper is a very recent work leveraging LSTM RNN. LSTM RNN is a more complicated form of RNN with a "forget gate". It allows reasonable good prediction of patient outcome given common missed or erretic medical data in EMR. It also claims the use of LSTM reduce the time to train, and require less data.

These result points us the potential of using LSTM RNN to analyze On-Q Trac and Sabre data.

----------

### Meeting #7, 12/13/16, SD

We discussed the following paper: 
[Semantic Segmentation using Adversarial Networks] (https://arxiv.org/abs/1611.08408)

The paper is proposing a new method for semantic segmentation using GAN. Instead of input noise, images are applied to the input of the generative model. The generative model will learn to produce a semantic mask of the image. The input to the discriminator is either the ground truth masks or the generated masks. The authors claim that they obtained better accuracy in terms of dice metric. The qualitative results show the improvement, reducing the false positives.

The group became creative and started to think of applications of GAN in other areas such as trading, lottery, etc :)

-----

### Meeting #6, 12/2/16, MA

We discussed the following paper: 
[Image-to-Image Translation with Conditional Adversarial Networks] (https://arxiv.org/abs/1611.07004)

We discussed how generative adversarial networks (GANs) work and how they can be applied to various problems including image generation, image segmentation. The idea of a general-purpose solution to image-to-image translation problems introduced in the paper is quite intersting.
TY mentioned that GANs might be even applicable to the patient data cleaning/interpretation problem.

The following video would be useful for better understanding of GANs [From Facebook AI research - Soumith Chintala - Adversarial Networks] (https://www.youtube.com/watch?v=QPkb5VcgXAM).


-------------

### Meeting #5, 11/18, SS

We watched and discussed the following video
[Deep Neural Networks in Medical Imaging and Radiology: Preventative and Precision Medicine Perspectives] (https://mygtc.gputechconf.com/events/35/schedules/3402)

This was one of the talks at GTC 2016 conference that SS attened. The talk is focused on employing deep learning for medical imaging applications. It has some intersting points. The speaker discusses how traditional image processing failed to combat challenges in medical imaging applications and how deep learning revolutionized the field. 

------

### Meeting #4, 11/09, JM

We discussed the following paper: [Clockwork Convnets for Video Semantic Segmentation] (https://arxiv.org/abs/1608.03609)

They try to incorporate temporal information from video’s into the convnet framework for segmentation. The frame-to-frame changes within the layers get smaller and smaller in deeper layers. The idea is that when there is not a lot of change in the video, some of the higher layer don't need to be calculated again but can be read from the cache.

Thomas also mentioned that key frames in mpeg4 could potentially be used as markers for change. They could indicate which frames are useful during the real-time deployment.

----------
### Meeting #3, 10/28, TY

Object segmentation from pump images. 

Interesting points: using different labels for each piece, using other objective functions such as binary_cross_entropy, using other image resolutions.


-----
### Meeting #2, 10/20, SD

I would like to talk about the work Facebook has been doing on Unsupervised edge generation
https://www.facebook.com/atscaleevents/videos/1682914828648281/
3D ConvNets , Unsupervised Edge generation using Optical Flow as a ground truth and then iterating to reinforce
 
Here is the paper that goes with the above Video
http://arxiv.org/pdf/1511.04166

Interesting points in this talk: using edge detection for pre-training, 3D convolutions for processing sequences of images(videos) for  different applications such as video classification and captioning, optical flow analysis

----------

###  Meeting #1, 10/12, MA

We discussed the following paper:
https://arxiv.org/abs/1511.07122

which proposes a new form of convolutional networks to reduce the number of parameters and improving accuracy.
The Github link is here:https://github.com/fyu/dilation

In addition, Dilated Convolution is now available in Keras: https://keras.io/layers/convolutional/#atrousconv2d
under: AtrousConvolution

--------------------------------



