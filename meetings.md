
# AI club summary


------
### First meeting, 10/12, MA

We discussed the following paper:
https://arxiv.org/abs/1511.07122

which proposes a new form of convolutional networks to reduce the number of parameters and improving accuracy.
The Github link is here:https://github.com/fyu/dilation

In addition, Dilated Convolution is now available in Keras: https://keras.io/layers/convolutional/#atrousconv2d
under: AtrousConvolution

-----
### Second meeting, 10/20, SD

I would like to talk about the work Facebook has been doing on Unsupervised edge generation
https://www.facebook.com/atscaleevents/videos/1682914828648281/
3D ConvNets , Unsupervised Edge generation using Optical Flow as a ground truth and then iterating to reinforce
 
Here is the paper that goes with the above Video
http://arxiv.org/pdf/1511.04166

Interesting points in this talk: using edge detection for pre-training, 3D convolutions for processing sequences of images(videos) for  different applications such as video classification and captioning, optical flow analysis


----------
### Third meeting, 10/28, TY

Object segmentation from pump images. 

Interesting points: using different labels for each piece, using other objective functions such as binary_cross_entropy, using other image resolutions.

------

### Forth meeting, 11/09, JM

We discussed the following paper: https://arxiv.org/abs/1608.03609

They try to incorporate temporal information from video’s into the convnet framework for segmentation. The frame-to-frame changes within the layers get smaller and smaller, the higher in the network the layer is. The idea is that when there is not a lot of change in the video, some of the higher layer don't need to be calculated again but can be read from cache.

Thomas also mentioned that key frames in mpeg4 could potentially be used as markers for change. They could indicate which frames are useful to train the AI.
