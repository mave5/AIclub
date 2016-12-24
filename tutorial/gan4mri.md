
## GANs for medical imaging.

In this work we aim to introduce a method of using genrative adversarial networks (GANs) to simultaneosly generate medical images and corresponding anntations. 

### Introduction
Automatic organ detection and segmentation have a huge impact in medical imaging applications. For instance in cardiac analysis, automatic segmentation of the heart chambers are used for cardiac volume and ejection fraction calculation. There is a good trend of employing deep learning algorithms in medical imaging. However, the main challenge in this field is the lack of data and annotations. Specifically, the annotations have to be performed by experts, which is costly and time-consuming. In this work, we introduce a method for simultaneous generation of data and annotations using GANs. 

#### Why this is important?
Considering the scarcity of data and annotations in medical imaging applications, the generated data and annotations using our method can be used for developing data-hungry deep learning algorithms. In addition, our method can be employed for anatomy detection.

### Data
We used the [MICCAI 2012 RV segmentation challenge dataset] (http://www.litislab.fr/?projet=1rvsc).
TrainingSet, including 16 patients with images and expert annotations, was used to develop the algorithm. We convert the annotations to binary masks with the same size as images. The original images/masks dimensions are 216 by 256. For tractable training, we downsampled the images/masks to 32 by 32. A sample image and corresponding annotation of the right ventricle (RV) of the heart is shown below.

![fogure] (https://github.com/mravendi/AIclub/blob/master/figs/realsample2.png) 


### Methods
The network has two blocks: 
* Generator: A convolutional neural network to generate images and corresponding masks.  
* Discriminator: A convolutional neural network to classify real images/masks from generated images/masks.

Here mask refers to a binary mask corresponding to the annotation. 

Block diagram of the network is shown below. ![Figure] (https://github.com/mravendi/AIclub/blob/master/figs/gan1.png)





#### Training algorithm:

1. Initialize Generator and Discriminator randomly.

2. Generate some images/masks using Generator.

3. Train Discriminator using the collected real images/masks (with y=1 as labels) and generated images/masks (with y=0 as labels).

4. Freeze the weights in Discriminator and stack it to Generator (figure below).

5. Train the stacked network using the generated images with y=1 as forced labels. 

6. Return to step 2.

It is noted that, initially, the generated images and masks are garbage. As training continious they become meaningful.  ![stacked network] (https://github.com/mravendi/AIclub/blob/master/figs/gan2.png).


You can see the code in [this jupyter notebook] (http://nbviewer.jupyter.org/github/mravendi/AIclub/blob/master/tutorial/notebook/GAN_CMRI_32by32.ipynb)



### References:
* [DCGAN](https://github.com/rajathkumarmp/DCGAN)
* [How to Train a GAN?](https://github.com/soumith/ganhacks)
* [KERAS-DCGAN](https://github.com/jacobgil/keras-dcgan)
* [Keras GAN](https://github.com/mravendi/KerasGAN)
* [Keras-GAN](https://github.com/phreeza/keras-GAN)
