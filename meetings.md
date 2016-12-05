
# AI club summary


------
###  Meeting #1, 10/12, MA

We discussed the following paper:
https://arxiv.org/abs/1511.07122

which proposes a new form of convolutional networks to reduce the number of parameters and improving accuracy.
The Github link is here:https://github.com/fyu/dilation

In addition, Dilated Convolution is now available in Keras: https://keras.io/layers/convolutional/#atrousconv2d
under: AtrousConvolution

-----
### Meeting #2, 10/20, SD

I would like to talk about the work Facebook has been doing on Unsupervised edge generation
https://www.facebook.com/atscaleevents/videos/1682914828648281/
3D ConvNets , Unsupervised Edge generation using Optical Flow as a ground truth and then iterating to reinforce
 
Here is the paper that goes with the above Video
http://arxiv.org/pdf/1511.04166

Interesting points in this talk: using edge detection for pre-training, 3D convolutions for processing sequences of images(videos) for  different applications such as video classification and captioning, optical flow analysis


----------
### Meeting #3, 10/28, TY

Object segmentation from pump images. 

Interesting points: using different labels for each piece, using other objective functions such as binary_cross_entropy, using other image resolutions.

------

### Meeting #4, 11/09, JM

We discussed the following paper: [Clockwork Convnets for Video Semantic Segmentation] (https://arxiv.org/abs/1608.03609)

They try to incorporate temporal information from video’s into the convnet framework for segmentation. The frame-to-frame changes within the layers get smaller and smaller in deeper layers. The idea is that when there is not a lot of change in the video, some of the higher layer don't need to be calculated again but can be read from the cache.

Thomas also mentioned that key frames in mpeg4 could potentially be used as markers for change. They could indicate which frames are useful during the real-time deployment.

------

### Meeting #5, 11/18, SS

We watched and discussed the following video
[Deep Neural Networks in Medical Imaging and Radiology: Preventative and Precision Medicine Perspectives] (https://mygtc.gputechconf.com/events/35/schedules/3402)

This was one of the talks at GTC 2016 conference that SS attened. The talk is focused on employing deep learning for medical imaging applications. It has some intersting points. The speaker discusses how traditional image processing failed to combat challenges in medical imaging applications and how deep learning revolutionized the field. 

-----

### Meeting #6, 12/2/16, MA

We discussed the following paper: 
[Image-to-Image Translation with Conditional Adversarial Networks] (https://arxiv.org/abs/1611.07004)

We discussed how generative adversarial networks (GANs) work and how they can be applied to various problems including image generation, image segmentation. The idea of a general-purpose solution to image-to-image translation problems introduced in the paper is quite intersting.
TY mentioned that GANs might be even applicable to the patient data cleaning/interpretation problem.

The following video is also useful to watch for understanding of GANs [From Facebook AI research - Soumith Chintala - Adversarial Networks] (https://www.youtube.com/watch?v=QPkb5VcgXAM).

----------




