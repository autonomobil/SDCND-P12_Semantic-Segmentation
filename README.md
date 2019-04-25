[//]: # (Image References)

[img1]: ./images/umm_000008.png "result1"
[img2]: ./images/umm_000014.png "result2"
[img3]: ./images/uu_000063.png "result3"
[img5]: ./images/um_000005.png "result4"
[img6]: ./images/um_000061.png "result5"
[img4]: ./images/structure.png "structure"

[img7]: ./images/uu_00009702.png "augmented1"
[img8]: ./images/uu_road_00009702.png "augmented2"
[img9]: ./images/uu_00009101.png "augmented3"
[img10]: ./images/uu_road_00009101.png "augmented4"
___

# SDCND Term 3 Project 12: Semantic Segmentation
## Semantic Segmentation Project for Udacity's Self-Driving Car Engineer Nanodegree Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, a Fully Convolutional Network (FCN) is used to label the pixels of a road in images. It is trained on the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

Some results:
![img1]
![img2]
![img3]
![img5]
![img6]

The following files were changed compared to the [seed project](https://github.com/udacity/CarND-Semantic-Segmentation):
*  `helper.py`
* `main.py`

These files were added:
*  `augment_data.py`
*  `helper2.py`


###Augmentation
The Kitti road dataset was augmented, by flipping, adding noise, zoom, warp, moving and colorshifting, see [augment_data.py](./augment_data.py).

Examples:
![img7] ![img8]
![img9] ![img10]

###Fully Convolutional Network (FCN)
####Structure
![img4]
#####Why Layer skip 3, 4 and 7?
In `main.py`, layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

####Training 

+ keep_prob: 0.45
+ learning_rate: 0.0001
+ epochs: 100
+ batch_size: 8
+ L2-Regularizer on each layer
+ Adam-Optimizer
+ Cross-entropy-loss




______________________________
####Notes: GPU
`main.py` will check to make sure a GPU is available


