<h1 align="center">
   FaceDetection using Viola-Jones' Robust Algorithm for Object Detection
</h1>


## Introduction

This is a Julia implementation of [Viola-Jones' Object Detection algorithm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.10.6807).  Although there is an [OpenCV port in Julia](https://github.com/JuliaOpenCV/OpenCV.jl), it seems to be ill-maintained.  As this algorithm was created for commercial use, there seem to be few widely-used or well-documented implementations of it on GitHub.  The implementation this repository is based off is [Simon Hohberg's Pythonic repository](https://github.com/Simon-Hohberg/Viola-Jones), as it seems to be well written (amd the most starred Python implementation on GitHub, though this is not necessarily a good measure), and Julia and Python alike are easy to read and write in &mdash; my thinking was that this would be easy enough to replicate in Julia, except for Pythonic classes, where I would have to use `struct`s (or at least easier to replicate from than, for example, [C++](https://github.com/alexdemartos/ViolaAndJones) or [JS](https://github.com/foo123/HAAR.js) &mdash; two other highly-starred repositories.).

I *implore* collaboration.  I am an undergraduate student with no formal education in computer science (or computer vision of any form for that matter), and I am pretending to write an algorithm knowing very little about the ideas behind the concepts.  There is a chance that I have done something incorrect, and I am certain this code can be refined by better programmers than myself.  Please, help me out if you like!

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

## Running the Algorithm

After you have setup your workspace, simply run
```bash
~/FaceDetection.jl/example.jl
```

## Timeline of Progression

 - [a79ab6f9](https://github.com/jakewilliami/FaceDetection.jl/commit/a79ab6f9) &mdash; Began working on the algorithm; mainly figuring out best way to go about this implementation.
 - [fd5e645c](https://github.com/jakewilliami/FaceDetection.jl/commit/fd5e645c) &mdash; First "Julia" adaptation of the algorithm; still a *lot* of bugs to figure out.
 - [2fcae630](https://github.com/jakewilliami/FaceDetection.jl/commit/2fcae630) &mdash; Started bug fixing using `src/FDA.jl` (the main example file).
 - [f1f5b5ea](https://github.com/jakewilliami/FaceDetection.jl/commit/f1f5b5ea) &mdash; Getting along very well with bug fixing (created a `struct` for Haar-like feature; updated weighting calculations; fixed `hstack` translation with nested arrays).  Added detailed comments on each function.
 - [a9e10eb4](https://github.com/jakewilliami/FaceDetection.jl/commit/a9e10eb4) &mdash; First working draft of the algorithm (without image reconstruction)!
 - [6b35f6d5](https://github.com/jakewilliami/FaceDetection.jl/commit/6b35f6d5) &mdash; Finally, the algorithm works as it should.  Just enhancements from here on out.


## To-do Notes

 - At the moment, `QuartzImageIO` is required to read `.jpg` files, but the programme is not set up to do so.  At the minute, this programme only accepts `.png` and `.pgm` files (thank's to `ImageIO` and `Netpbm` respectively).
 - I am unsure at the moment whether this programme will work if the test images are of differing dimensionality to the training faces.  I suspect it won't, but I will test this soon.

## Acknowledgements

Thank you to:

 - [**Simon Honberg**](https://github.com/Simon-Hohberg) for the original open-source Python code upon which this repository is largely based.  This has provided me with an easy-to-read and clear foundation for the Julia implementation of this algorithm;
 - [**Michael Jones**](https://www.merl.com/people/mjones) for (along with [Tirta Susilo](https://people.wgtn.ac.nz/tirta.susilo)) suggesting the method for a *facelike-ness* measure;
 - [**Mahdi Rezaei**](https://environment.leeds.ac.uk/staff/9408/dr-mahdi-rezaei) for helping me understand the full process of Viola-Jones' object detection;
 - [**Ying Bi**](https://ecs.wgtn.ac.nz/Main/GradYingBi) for always being happy to answer questions (which mainly turned out to be a lack of programming knowledge rather than conceptual);
 - [**David Peck**](https://github.com/dmipeck) for being ever so patient with my lack of understanding of programming, and helping me with conceptual translation from Python to Julia;
 - Finally, the people in the Julia slack channel, for dealing with many (probably stupid) questions.  To name a few: Micket, David Sanders, Eric Forgy, Jakob Nissen, and Roel.
