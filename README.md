<h1 align="center">
   FaceDetection using Viola-Jones' Robust Algorithm for Object Detection
</h1>

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jakewilliami.github.io/FaceDetection.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jakewilliami.github.io/FaceDetection.jl/dev)
[![CI](https://github.com/jakewilliami/FaceDetection.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jakewilliami/FaceDetection.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
![Project Status](https://img.shields.io/badge/status-maturing-green)

---

### Caveats;&mdash; help wanted

Currently, there are two main areas I would love some help with:
  1. [Reading](https://github.com/jakewilliami/FaceDetection.jl/issues/20) from [pre-trained](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) data files would greatly help those who are not training the object detection framework themselves; and
  2. Creating ([*scalable*](https://github.com/jakewilliami/FaceDetection.jl/issues/21)) [bounding boxes around detected objects](https://github.com/jakewilliami/FaceDetection.jl/issues/6).  Everyone loves a visual; you can show your grandmother a face with a box around it, but she'll switch off if you show her any code.  People want pictures!

Since this project helped me with the `get_faceness` function (this idea of scoring the "faceness" of an image), now that I have data from this I have put this project [on the back burner](https://dictionary.cambridge.org/dictionary/english/on-the-back-burner).  However, in the interest of others, if anyone were to help with these two items (and any other [issues](https://github.com/jakewilliami/FaceDetection.jl/issues/)), they would forever have my gratitude.

---

## Introduction

This is a Julia implementation of [Viola-Jones' Object Detection algorithm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.10.6807).  Although there is an [OpenCV port in Julia](https://github.com/JuliaOpenCV/OpenCV.jl), it seems to be ill-maintained.  As this algorithm was created for commercial use, there seem to be few widely-used or well-documented implementations of it on GitHub.  The implementation this repository is based off is [Simon Hohberg's Pythonic repository](https://github.com/Simon-Hohberg/Viola-Jones), as it seems to be well written (and the most starred Python implementation on GitHub, though this is not necessarily a good measure). Julia and Python alike are easy to read and write in &mdash; my thinking was that this would be easy enough to replicate in Julia, except for Pythonic classes, where I would have to use `struct`s (or at least easier to replicate from than, for example, [C++](https://github.com/alexdemartos/ViolaAndJones) or [JS](https://github.com/foo123/HAAR.js) &mdash; two other highly-starred repositories.).

I *implore* collaboration.  I am an undergraduate student with no formal education in computer science (or computer vision of any form for that matter); I am certain this code can be refined/optimised by better programmers than myself.  Please, help me out if you like!

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

## Quick Start

```julia-repl
using FaceDetection

# Constants
pos_training_path, neg_training_path = "...", "..."
num_faces, num_non_faces = length(filtered_ls(pos_testing_path)), length(filtered_ls(neg_testing_path))  # You can also just put in a simple number here, if you know how many training images you have
num_classifiers = 10
min_feature_height, max_feature_height = 0, 19
min_feature_width, max_feature_width = 0, 19
scale, scale_to = true, (19, 19)  # we want all training images to be standardised to size 19x19

# Train a model
classifiers = FaceDetection.learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = scale, scale_to = scale_to)

# Results
correct_faces = sum(ensemble_vote_all(pos_testing_path, classifiers, scale=scale, scale_to=scale_to))
correct_non_faces = num_non_faces - sum(ensemble_vote_all(neg_testing_path, classifiers, scale=scale, scale_to=scale_to))
println((correct_faces / num_faces) * 100, "% of faces were recognised as faces")
println((correct_non_faces / num_non_faces) * 100, "% of non-faces were identified as non-faces")
```

For more examples like this, see [`examples/`](examples/).

## Citation

If your research depends on FaceDetection.jl, please consider giving us a formal citation: [`citation.bib`](./citation.bib).

## Miscellaneous Notes

### Timeline of Progression

 - [a79ab6f9](https://github.com/jakewilliami/FaceDetection.jl/commit/a79ab6f9) &mdash; Began working on the algorithm; mainly figuring out best way to go about this implementation.
 - [fd5e645c](https://github.com/jakewilliami/FaceDetection.jl/commit/fd5e645c) &mdash; First "Julia" adaptation of the algorithm; still a *lot* of bugs to figure out.
 - [2fcae630](https://github.com/jakewilliami/FaceDetection.jl/commit/2fcae630) &mdash; Started bug fixing using `src/FDA.jl` (the main example file).
 - [f1f5b5ea](https://github.com/jakewilliami/FaceDetection.jl/commit/f1f5b5ea) &mdash; Getting along very well with bug fixing (created a `struct` for Haar-like feature; updated weighting calculations; fixed `hstack` translation with nested arrays).  Added detailed comments on each function.
 - [a9e10eb4](https://github.com/jakewilliami/FaceDetection.jl/commit/a9e10eb4) &mdash; First working draft of the algorithm (without image reconstruction)!
 - [6b35f6d5](https://github.com/jakewilliami/FaceDetection.jl/commit/6b35f6d5) &mdash; Finally, the algorithm works as it should.  Just enhancements from here on out.
 - [854bba32](https://github.com/jakewilliami/FaceDetection.jl/commit/854bba32) and [655e0e14](https://github.com/jakewilliami/FaceDetection.jl/commit/655e0e14) &mdash; Implemented facelike scoring and wrote score data to CSV (see [#7](https://github.com/jakewilliami/FaceDetection.jl/issues/7)).
 - [e7295f8d](https://github.com/jakewilliami/FaceDetection.jl/commit/e7295f8d) &mdash; Implemented writing training data to file and reading from that data to save computation time.
 - [e9116987](https://github.com/jakewilliami/FaceDetection.jl/commit/e9116987) &mdash; Changed to sequential processing.
 - [750aa22d](https://github.com/jakewilliami/FaceDetection.jl/commit/750aa22d)&ndash;[b3aec6b8](https://github.com/jakewilliami/FaceDetection.jl/commit/b3aec6b8) &mdash; Optimised performance.
 <!---[]() &mdash;--->

### Acknowledgements

Thank you to:

 - [**Simon Honberg**](https://github.com/Simon-Hohberg) for the original open-source Python code upon which this repository is largely based.  This has provided me with an easy-to-read and clear foundation for the Julia implementation of this algorithm;
 - [**Michael Jones**](https://www.merl.com/people/mjones) for (along with [Tirta Susilo](https://people.wgtn.ac.nz/tirta.susilo)) suggesting the method for a *facelike-ness* measure;
 - [**Mahdi Rezaei**](https://environment.leeds.ac.uk/staff/9408/dr-mahdi-rezaei) for helping me understand the full process of Viola-Jones' object detection;
 - [**Ying Bi**](https://ecs.wgtn.ac.nz/Main/GradYingBi) for always being happy to answer questions (which mainly turned out to be a lack of programming knowledge rather than conceptual; also with help from [**Bing Xue**](https://homepages.ecs.vuw.ac.nz/~xuebing/index.html));
 - **Mr. H. Lockwood** and **Mr. D. Peck** are Comp. Sci. students who have answered a few questions of mine;
 - Finally, the people in the Julia slack channel, for dealing with many (probably stupid) questions.  Just a few who come to mind: Micket, David Sanders, Eric Forgy, Jakob Nissen, and Roel.

### A Note on running on BSD:

The default JuliaPlots backend `GR` does not provide binaries for FreeBSD.  [Here's how you can build it from source.](https://github.com/jheinen/GR.jl/issues/268#issuecomment-584389111).  That said, `StatsPlots` is only a dependency for an example, and not for the main package.


[code-style-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[code-style-url]: https://github.com/invenia/BlueStyle
