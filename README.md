Since the class pivoted away from the Medium articles style of teaching, this project will not be completed.

In Development: Triplet Loss Network in Keras (with TensorFlow backend)
=======================================================
A quick implementation of a triplet network with an online (batch-based) triplet
loss (in Keras).

This was primarily written to support a series of Medium articles ("Building a 
Triplet Network in Keras") written as a tutorial for students in UMD's FIRE171 
(and FIRE271) class who are working with similarity learning. These are the 
article subtitles:
1. [Building the Embedding Network](https://medium.com/@tmthylin/building-a-triplet-network-in-keras-part-i-f13e7d711e1b)
2. [Customizing the Triplet Loss Function]
3. [Training the Model]
4. [Evaluating and Visualizing Results]

More implementation details are in the articles.

Data
----
The data used for training is a super small (~140 images) dataset of a few
families of bees, collected from manual Google Images search. The dataset I used 
is provided in the release. I currently only have a training set (I need more 
time to collect validation and test set).

All images are resized and padded to 299x299 (padding with repeats, using the 
default numpy resize function).

Model
-----
The model is simple: I chose to use InceptionV3 as the backbone here since it
provides high accuracy while not being super large. Plus, for the purposes of 
the articles, there is already a prebuilt Keras implementation with weights 
pretrained on ImageNet. The last fully-connected layer is removed, and we add a
few new fully-connected layers for our output embedding.

Loss Function
-------------
The version of triplet loss that is used is the online version (no prior triplet
mining). This is implemented in TensorFlow. This is for a few reasons:
* higher accuracy with faster training time (see FaceNet paper)
* the training model and the inference model are the same
* we can care less about which images go into each batch
* no need to store anything about the triplets beforehand
* this is the version they are expected to use for their student projects

However, it poses some challenges/constraints:
* we need to make sure there's at least one positive pair in the batch
* the loss function itself is more complicated to write (we have to find pairs using the 
TensorFlow API only)


Training
--------

Evaluation
----------

Visualization
-------------
