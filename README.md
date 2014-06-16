Image Categorization Application
===
## Application
* Coding Language: Matlab.
* Image Feature: SIFT(Scale-Invariant Feature Transform).
* Sparse Coding: LLC(Locality-constrained Linear Coding).
* Classifier: SVM + Bagging.

## About
* This is an example code for the algorithm described in

Jinjun Wang, Jianchao Yang, Kai Yu, Fengjun Lv, Thomas Huang and Yihong Gong.
"Locality-constrained Linear Coding for Image Classification", CVPR 2010.

* To train the codebook, one can simply use K-means. A pre-trained codebook is included in the package for Caltech101.

* For SIFT descriptor extraction, we use Prof. Lazebnik's matlab codes.

* We use bagging method to improve machine learning (SVM) accuracy.

* "image" directory stores all images, "data" directory stores all SIFT features, "features" directory stores all LLC features.

## Usage
* To run the example code: 
1. Put Caltech101 image dataset in the "image" directory and rename the folder with "Caltech101". You can use other dataset as well.
2. Start from LLC_Test.m