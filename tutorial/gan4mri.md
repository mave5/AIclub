
## GANs for medical imaging.

In this work we aim to introduce a method of using genrative adversarial networks (GANs) to simultaneosly generate medical images and corresponding anntations.

### Introduction
Automatic organ detection and segmentation can have a huge impact in medical imaging applications. For instance in cardiac analysis, automatic segmentation of the heart chambers can be used for cardiac volume and ejection fraction calculation. There is a good trend of employing deep learning algorithms in medical imaging. However, the main challenge in this field is the lack of data and annotations. Specifically, the annotations has to be done by experts, which is costly and time-consuming. 

### Methods
The network has two blocks: 
* Generator: A convolutional neural network to generate image and corresponding mask.  
* Discriminator: A convolutional neural network to classify real images/mask from generated images/masks.

Here mask refers to a binary mask corresponding to the annotation. 

Block diagram of the network is shown below. ![Figure] (https://github.com/mravendi/AIclub/blob/master/figs/gan1.png)

Training algorithms:
1. Initialize Generator and Discriminator randomly.

2. Generate some images/masks using Generator.

3. Train Discriminator using the collected real images/masks (with y=1 as labels) and generated images/masks (with y=0 as labels).

4. Stack Generator and Discriminator together.

5. Train the stacked network using the generated images with y=1 as forced labels. 

6. Repeat to step 2.

It is noted that, initially, the generated images and masks are garbage. But as training continious they become meaningful. 





