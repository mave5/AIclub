
# AIclub meeting summary
--------------


### Meeting #28, 1/9/2017, MA

We watched Andrew Ng's talk at [EMTech](https://www.youtube.com/watch?v=NKpuX_yzdYs).

He points out:
- AI is the new electricity
- %98 of current AI values in indusrty comes from supervised learning
- as a rule of thumb, anyhting that a person can do in less than a second can be automated using AI
- a lot of jobs are sequences of 1-second tasks, such security type of work, at risk of automation
- plot of performance vs data size for various ML algorithms
- current AI algorithms
  -- supervised learning: hunger for data
  -- transfer learning
  -- unsupervised learning
  -- reinforcement learning: even more hunger for data and away from large scale deployment
-  web search example: data asset is key
- data -> product -> users -> more data -> improve system
- example: rise of  internet
- a shopping mall + internet is not an internet company
    -- A/B testing
    -- short cycle times
    -- decision making push to pm and engs
 - rise of AI era
 - a tech company + NN is not an AI company
 - strategic data acqusition
 - unified data warehouses
 - pervasive automation
 - new job describtion
 - how to integrate AI into your tech company
 - building a team of AI on the side to help other divisions
 


----------------
### Meeting #27, 11/10/2017, MA

We discussed the following paper:
[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)

They introduce a method for data augmentation called random erasing.
Random erasing randomly selects a rectangle region in an image and erases its pixels with random values.
Random erasing seem to be an easy and effective idea for data augmentation.
It shows improvement in classification accuracy of multiple networks. The deeper the network, the more effective.
Seems to be not so sensetive to the probability of erasing.
They compared their results with drop out and seems to outperform dropout.

--------------------
### Meeting #26, 8/11/2017, MA

We wathced a cool video on q-learning and discussed a few slides on Reinforcement Learning. We are going to continue this topic in the future.

----------------
### Meeting #25, 7/27/2017, YM

We discussed article [SonoNet: Real-Time Detection and Localisation of Fetal Standard Scan Planes in Freehand Ultrasound](https://arxiv.org/pdf/1612.05601.pdf)

This is a pretty thorogh article on using deep learning for fetus detection, localization and analysis in ultrasound imaging.
It talks in detail all the procedure from data collection, annotation and training.
The article also proposes a semi-supervised method for ground-truth generation using the extracted features in the deeper layers of a classification network.


---------------------
### Meeting #24, 7/14/2017, HY

We discuss article [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)

- prunning, quantization, Huffman coding
- to be deployed on mobile devices, energy consumption
- fit SRAM (inside chip) vs DRAM (external ram) 
- train with full precision, prunn, re-train
- using code book, compression, weight sharing
- using k-means to clustering
- using customize Caffe
- compress rate 


We also discussed [How to quantize neural networks with Tensorflow](https://www.tensorflow.org/performance/quantization)
- first floating point then quantize


------------------
#### Meeting #23, 7/13/2017, MA
We watched two videos from Rework Deep Learning in healthcare, Boston 2017.

[DL Rework Boston 2017](http://videos.re-work.co/events/24-deep-learning-in-healthcare-summit-boston-2017)

* Reducing the Tedium of Radiology with Cloud-based Deep Learning Daniel Golden, Director of Machine Learning at Arterys
  At some points he mentions to custom loss function to be able to use all of training data.
  
* Dental CAD/CAM Automation with Deep Learning Sergei Azernikov, Machine Learning Lead at Glidewell Laboratories


---------------------------
#### Meeting #22, 5/19/2017, SS

we reviewed and discussed the following articles
- [Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x)
- [A comparison of ground truth estimation methods](http://link.springer.com/article/10.1007%2Fs11548-009-0401-3)
- [Segmentation Validation Framework, Ghazaleh Safarzadeh Khooshabi](http://www.uppsatser.se/om/Ghazaleh+Safarzadeh+Khooshabi/)


 Summary
  - Different metrics used for accuracy measurement in image segmentation 
    - overlap (dice), distance(HD), probabilistic
  - Different methods (algorithms) used for ground truth generation  
  - Williams idex to cluster/rank different annotators
  - itk, vtk software tools
  - different methods for building ground truth: averaging, staple method



-----------------------
### Meeting # 21, 5/5/2017, MA

We watched this video from Facebook developer 2017 (F8) conference
[Delivering Real-Time AI In the Palm of Your Hand](https://developers.facebook.com/videos/f8-2017/delivering-real-time-ai-in-the-palm-of-your-hand/)

- talk is about Caffe2, a deep learning framework. Caffe2 is the extension of Caffe1 library with the goal of building a cross-platform framework. 
- the talk does not go into details so it is hard to get any insights.

--------------------------
### Meeting #20, 4/21/2017, SD

We discussed article by Arterys [FastVentricle: Cardiac Segmentation with ENet](https://arxiv.org/abs/1704.04296)
Arterys have FDA approval of a deep learning SW tool (DeepVentricle) for cardiac MRI. This article describes a faster version namely FastVentricle.

Interesting points from the article
- using ENET optimized for speed and memory
- databse of 1143 short-axis cine dataset, 80% for train and 10% for validation, 10% as hold-out test
- 10% hold out only from licensed radiologists
- using random hyper-parameter search
- internal representation using deep-dream (google article)
- using post-processing for difficult cases (congenital heart disease)


---------------------------
### Meeting # 19, 3/31/2017, MA

We discussed article [Who Said What: Modeling Individual Labelers Improves
Classification](https://arxiv.org/pdf/1703.08774.pdf)

This is a nice article from Hinton's group. 
They use the same dataset as in this [Development and Validation of a Deep Learning Algorithm
for Detection of Diabetic Retinopathy
in Retinal Fundus Photographs](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45732.pdf) for their experiments.

They argue that expert annotators have different level of reliability. As a result, the classical majority voting from multiple experts for creating the ground truth might not be the best approach. 
Instead, they propose modeling the doctors annotations by adding an extra output per doctor on top of Inception-v3 features.
At inference time, the outcome of modeled doctors could be averaged equally or using learnable weighted average.
The result show improvement in the performance compare to the classical averaging of annotations.

The summray of slides can be found [here](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/March312017.pdf).

-----------

### Meeting #18, 3/24,2017, JM

We discussed paper: [Mask R-CNN](https://arxiv.org/abs/1703.06870).
This is work on top of two other articles Fast R-CNN and Faster R-CNN, which use region proposals for object detection.
In Mask R-CNN, a new branch is added to provide object segmentation. While extracting an ROI and then performing object segmentation would lead to a better performance, it seems to be computationally expensive. Anyway, the provided results are really good, comparable to ground truth!


------------

### Meeting #17, 3/10/2017, TY

TY presented how to deploy Keras models in iPad.

---------

### Meeting #16, 2/21/2017, MA, SD

We did a lunch and learn presentation on article:
[Dermatologist-level classification of skin cancer with deep neural networks](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html)

It uses a pre-trained Inception-V3 network fine-tuned on skin cancer images able to classify 2000 different diseases.

The slides are [here](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/AIClub_LunchAndLearn_Feb2017.pdf)


-------------

### Meeting #15, 2/17/2017, MA

We first watched and discussed this [video](https://www.youtube.com/watch?v=UeheTiBJ0Io&list=PLOU2XLYxmsIKGc_NBoIhTn2Qhraji53cv&index=7)
presented by Francois Chollet (creator of Keras) at TensorFlow Dev Summit 2017. He explains how Keras is going to be integrated into Tensorflow and also provides a concrete example on using Keras for video captioning.

We also discussed paper [Convolutional Gated Recurrent Networks for Video Segmentation](https://arxiv.org/abs/1611.05435).
This paper proposes a combination of fully-convolutional neural networks (F-CNN) with recurrent neural nets (RNN/GRU/LSTM) for video segmentation. Previous methods rely on single frame segmentation, while extracting the correlation between frames using RNN could lead to improvement in the outcome. Their results show significant boost specially for smaller objects.
Slides are [here.](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/rcnn_seg_feb_2017.pdf)


-------------
### Meeting #14, 2/10/17, JM

We discussed paper [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)
The paper argues that the reason behind generalization capability of neural networks should be reconsidered! They provided some experiments and results that are interesting! In an experiment, they train several standard NN architectures on a copy of the data where the true labels were replaced by random labels. Based on this experiment, they summarized that:
- Deep neural networks easily fit random labels.
- Capacity of neural networks is large enough for a brute force memorization.
- Training time on random labels increases only by a small constanct factor compared with training with true labels.

-------------

### Meeting #13, 2/3/17, TY

TY attended ReWORK deep learning conference 2017 in SF and gave a presentation on the conference.
Slides can be found [here.](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/Re-work%20summary_2017.pdf)

--------

### Meeting #12, 1/27/17, MA
We reviewed paper: [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)
The paper proposes using 3D CNN (as opposed to 2D CNN) for extracting temporal information between videos frames for Video analysis.

The paper claims are summarized as:

- 3D ConvNets are more suitable for spatiotemporal feature learning compared to 2D ConvNets.
- A homogeneous architecture with small 3*3*3 convolution kernels in all layers is among the best performing architectures for 3D ConvNets.
- Our learned features, namely C3D (Convolutional 3D), with a simple linear classifier outperform state-of-the-art methods on 4 different benchmarks and are comparable with current best methods on the other 2 benchmarks. 

The slides can be found [here](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/conv3d_jan27.pdf). 

-----

### Meeting #11, 1/20/17, SD

We discussed the following HW , Paper & Videos:
[Googles Project Soli](https://atap.google.com/soli/)

Paper [Soli: ubiquitous gesture sensing with millimeter wave radar](http://dl.acm.org/citation.cfm?id=2925953)

[Soli Overview](https://www.youtube.com/watch?v=0QNiZfSsPc0)

[Google I/O Talk](https://www.youtube.com/watch?v=czJfcgvQcNA)

[Object Recognition](http://www.digitaltrends.com/computing/radarcat-machine-learning-radar-sensor-google-soli/)

[Infineon Datasheet](http://www.infineon.com/dgdl/FAQ+Document+Infineon+Google+Soli.pdf?fileId=5546d46156e151000156e4c0f0b50001)

The Paper covers the design of a hand gesture recognition system using novel combination of hardware and software processing pipelines. The paper discusses the benifits of keeping the software modular in a processing pipeline and using different feature modeling and extraction before training a machine learning model ontop of these features. 

There is another paper: [Interacting with Soli: Exploring Fine-Grained Dynamic Gesture Recognition in the Radio-Frequency Spectrum](http://dl.acm.org/citation.cfm?id=2984565), that processes signals from soli with recurrent CNN for accuracte gesture recognition.

-----

### Meeting #10, 1/12/17, MA

We discussed the following paper:
[Learning From Noisy Large-Scale Datasets With Minimal Supervision](https://arxiv.org/abs/1701.01619)

The paper introduces a method for classification networks to take advantage of a large but noisy annotations to improve their accuracy.
In this work, a dataset with 9M images with noisy annotations and only 40K images with cleaned annotations are used. The idea is to train a network to learn the noise pattern and then use the network to clean the noisy annotations. Then both the clean data and the reduced-noise data are used for training. The authors claim that their approach is more effective than the traditional pre-training with noisy data and then fine-tunning with the clean data. Their results supports the claim. 

The slides can be found [here](https://github.com/mravendi/AIclub/blob/master/tutorial/presentations/learningfromnoisydata.pdf).

---------------

### Meeting #9, 1/6/17, JM

We discussed the following paper:
[Intriguing properties of neural networks](https://arxiv.org/pdf/1312.6199v4)

The paper discusses 2 somewhat surprising properties of neural networks:
1. The idea that a particular neuron in the higher layers of the network stores specific semantic information is too simplistic. It turns out that semantic information is encoded in a many neurons together.
2. Using the gradients from a forward pass, it is possible to calculate the changes that need to be made to an image in order for it to be miss-classified with very high probability (adverserial example). These changes turn out to be surprisingly small (not visible) and they also seem to generalize to a worrying degree to across both different architectures and networks trained on different data. 

In the discussion we thought that GANs might help for the network to become more robust against adversarial examples. The authors seemed to have gone more into GAN research. Good topic for future meetings.

 --------------
 
### Meeting #8, 12/21/16, TY

We discussed the following three papers:
[A Comparison of Models for Predicting Early Hospital Readmission](http://www.sciencedirect.com/science/article/pii/S1532046415000969)
[Dr. AI: Predicting Clinical Events via Recurrent Neural Network](https://arxiv.org/pdf/1511.05942v11.pdf)
[Learning to Diagnose with LSTM Recurrent Neural Network](https://arxiv.org/pdf/1511.03677v6.pdf) 

These three papers show a series of effort to use Machine Learning to predict patient outcome over existing public or private EMR data. These EMR data serve a good secondary use in Machine Learning to drive down healthcare cost. The first paper shows using a simple RBM consistently out-perform traditional statistica methods to predict the chance of hospital readmission. It does not explore time sequence medical data because RBM lacks the necessary construct.

The second paper is the first use or RNN with GRU over public patient data. It shows promising result of exploring the time sequence medical data in predicting the future diagnosis.

The third paper is a very recent work leveraging LSTM RNN. LSTM RNN is a more complicated form of RNN with a "forget gate". It allows reasonable good prediction of patient outcome given common missed or erretic medical data in EMR. It also claims the use of LSTM reduce the time to train, and require less data.

These result points us the potential of using LSTM RNN to analyze On-Q Trac and Sabre data.

----------

### Meeting #7, 12/13/16, SD

We discussed the following paper: 
[Semantic Segmentation using Adversarial Networks](https://arxiv.org/abs/1611.08408)

The paper is proposing a new method for semantic segmentation using GAN. Instead of input noise, images are applied to the input of the generative model. The generative model will learn to produce a semantic mask of the image. The input to the discriminator is either the ground truth masks or the generated masks. The authors claim that they obtained better accuracy in terms of dice metric. The qualitative results show the improvement, reducing the false positives.

The group became creative and started to think of applications of GAN in other areas such as trading, lottery, etc :)

-----

### Meeting #6, 12/2/16, MA

We discussed the following paper: 
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

We discussed how generative adversarial networks (GANs) work and how they can be applied to various problems including image generation, image segmentation. The idea of a general-purpose solution to image-to-image translation problems introduced in the paper is quite intersting.
TY mentioned that GANs might be even applicable to the patient data cleaning/interpretation problem.

The following video would be useful for better understanding of GANs [From Facebook AI research - Soumith Chintala - Adversarial Networks](https://www.youtube.com/watch?v=QPkb5VcgXAM).


-------------

### Meeting #5, 11/18, SS

We watched and discussed the following video
[Deep Neural Networks in Medical Imaging and Radiology: Preventative and Precision Medicine Perspectives](https://mygtc.gputechconf.com/events/35/schedules/3402)

This was one of the talks at GTC 2016 conference that SS attened. The talk is focused on employing deep learning for medical imaging applications. It has some intersting points. The speaker discusses how traditional image processing failed to combat challenges in medical imaging applications and how deep learning revolutionized the field. 

------

### Meeting #4, 11/09, JM

We discussed the following paper: [Clockwork Convnets for Video Semantic Segmentation](https://arxiv.org/abs/1608.03609)

They try to incorporate temporal information from video’s into the convnet framework for segmentation. The frame-to-frame changes within the layers get smaller and smaller in deeper layers. The idea is that when there is not a lot of change in the video, some of the higher layer don't need to be calculated again but can be read from the cache.

Thomas also mentioned that key frames in mpeg4 could potentially be used as markers for change. They could indicate which frames are useful during the real-time deployment.

----------
### Meeting #3, 10/28, TY

Object segmentation from pump images. 

Interesting points: using different labels for each piece, using other objective functions such as binary_cross_entropy, using other image resolutions.


-----
### Meeting #2, 10/20, SD

I would like to talk about the work Facebook has been doing on [Unsupervised edge generation](https://www.facebook.com/atscaleevents/videos/1682914828648281/)
3D ConvNets , Unsupervised Edge generation using Optical Flow as a ground truth and then iterating to reinforce
 
Here is the paper that goes with the above [Video](http://arxiv.org/pdf/1511.04166)

Interesting points in this talk: using edge detection for pre-training, 3D convolutions for processing sequences of images(videos) for  different applications such as video classification and captioning, optical flow analysis

----------

###  Meeting #1, 10/12, MA

We discussed the following [paper](https://arxiv.org/abs/1511.07122)

which proposes a new form of convolutional networks to reduce the number of parameters and improving accuracy.
The Github link is [here](https://github.com/fyu/dilation)

In addition, Dilated Convolution is now available in [Keras](https://keras.io/layers/convolutional/#atrousconv2d)
under: AtrousConvolution

--------------------------------



