
## GANs for medical imaging.

In this work we aim to introduce a method of using genrative adversarial networks (GANs) to simultaneosly generate medical images and corresponding anntations.

### Introduction
Automatic organ detection and segmentation can have a huge impact in medical imaging applications. For instance in cardiac analysis, automatic segmentation of the heart chambers can be used for cardiac volume and ejection fraction calculation. There is a good trend of employing deep learning algorithms in medical imaging. However, the main challenge in this field is the lack of data and annotations. Specifically, the annotations has to be done by experts, which is costly and time-consuming. 

### Methods
We employ a GAN with two blocks: 
* Generator: A convolutional neural network to generate image and corresponding annotation.  
* Discriminator: A convolutional neural network to classify real images/annotations from generated images/annotations. 





