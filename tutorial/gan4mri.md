
## Generating images and annotations for medical imaging analysis using Generative Adversarial Networks (GANs). 

Automatic organ detection and segmentation can have a main impact in medical imaging applications. For instance in cardiac analysis,  the annotations can be used for cardiac volume and ejection fraction calculation. There is good trend of employing deep learning algorithms in medical imaging applications. However, the main issue in this field is the lack of data and annotations. Specifically, the annotations has to be done by costly experts. 

This work aims to introduce a method of using GANs to simultaneosly generate cardiac images and corresponding anntations.

### Method
We employ a GAN with two blocks: 
* Generator: A convolutional neural network to generate image and corresponding annotation.  
* Discriminator: A convolutional neural network to classify real images/annotations from generated images/annotations. 



