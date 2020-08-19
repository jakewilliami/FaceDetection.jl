# FaceDetection using Viola-Jones' Robust Algorithm for Object Detection

## IMPORTANT NOTE:
**This repository is currently under construction.  There is no working algorithm yet.**

## Introduction

This is a Julia implementation of [Viola-Jones' Object Detection algorithm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.10.6807).  Although there is an [OpenCV port in Julia](https://github.com/JuliaOpenCV/OpenCV.jl), it seems to be ill-maintained.  As this algorithm was created for commercial use, there seem to be few widely-used or well-documented implementations of it on GitHub.  The implementation this repository is based off is [Simon Hohberg's Pythonic repository](https://github.com/Simon-Hohberg/Viola-Jones), as it seems to be well written (amd the most starred Python implementation on GitHub, though this is not necessarily a good measure), and Julia and Python alike are easy to read and write in &mdash; my thinking was that this would be easy enough to replicate in Julia, except for Pythonic classes, where I would have to use `struct`s (or at least easier to replicate from than, for example, [C++](https://github.com/alexdemartos/ViolaAndJones) or [JS](https://github.com/foo123/HAAR.js) &mdash; two other hgihly-starred repositories.).


## Installation and Set Up

Run the following in terminal:
```bash
cd ${HOME}
git clone https://github.com/jakewilliami/FaceDetection.jl.git/
cd FaceDetection.jl
bash setup.sh
```

## How it works

In an over-simplified manner, the Viola-Jones algorithm has some four stages:

 1. Takes an image, converts it into an array of intensity values (i.e., in grey-scale), and constructs an [Integral Image](https://en.wikipedia.org/wiki/Summed-area_table), such that for every element in the array, the Integral Image element is the sum of all elements above and to the left of it.  This makes calculations easier for step 2.
 2. Finds [Haar-like Features](https://en.wikipedia.org/wiki/Haar-like_feature) from Integral Image.
 3. There is now a training phase using sets of faces and non-faces.  This phase uses something called Adaboost (short for Adaptive Boosting).  Boosting is one method of Ensemble Learning. There are other Ensemble Learning methods like Bagging, Stacking, &c.. The differences between Bagging, Boosting, Stacking are:
      - Bagging uses equal weight voting. Trains each model with a random drawn subset of training set.
      - Boosting trains each new model instance to emphasize the training instances that previous models mis-classified. Has better accuracy comparing to bagging, but also tends to overfit.
      - Stacking trains a learning algorithm to combine the predictions of several other learning algorithms.
  Despite this method being developed at the start of the century, it is blazingly fast compared to some machine learning algorithms, and still widely used.
 4. Finally, this algorithm uses [Cascading Classifiers](https://en.wikipedia.org/wiki/Cascading_classifiers) to identify faces.  (See page 12 of the original paper for the specific cascade).
 
For a better explanation, read [the paper from 2001](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.10.6807), or see [the Wikipedia page](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) on this algorithm.
