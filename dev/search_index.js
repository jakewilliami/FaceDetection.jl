var documenterSearchIndex = {"docs":
[{"location":"resources/#Face-detection-resources/datasets","page":"Other Resources","title":"Face detection resources/datasets","text":"","category":"section"},{"location":"resources/","page":"Other Resources","title":"Other Resources","text":"# datasets\nhttps://github.com/INVASIS/Viola-Jones/ # main training dataset\nhttps://github.com/OlegTheCat/face-detection-data # alt training dataset\nhttp://cbcl.mit.edu/projects/cbcl/software-datasets/faces.tar.gz # MIT dataset\nhttp://tamaraberg.com/faceDataset/originalPics.tar.gz # FDDB dataset\nhttp://vis-www.cs.umass.edu/lfw/lfw.tgz # LFW dataset\nhttps://github.com/opencv/opencv/ # pre-trained models exist here\nhttps://github.com/jian667/face-dataset\n\n# resources\nhttps://github.com/betars/Face-Resources\nhttps://www.wikiwand.com/en/List_of_datasets_for_machine-learning_research#/Object_detection_and_recognition\nhttps://www.wikiwand.com/en/List_of_datasets_for_machine-learning_research#/Other_images\nhttps://www.face-rec.org/databases/\nhttps://github.com/polarisZhao/awesome-face#-datasets","category":"page"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Modules = [FaceDetection]","category":"page"},{"location":"usage/#Main.FaceDetection.HaarLikeObject","page":"Usage","title":"Main.FaceDetection.HaarLikeObject","text":"mutable struct HaarLikeObject{I <: Integer, F <: AbstractFloat}\n\n    Struct representing a Haar-like feature.\n    \nfeature_type::Tuple{I, I}\nposition::Tuple{I, I}\ntop_left::Tuple{I, I}\nbottom_right::Tuple{I, I}\nwidth::I\nheight::I\nthreshold::I\npolarity::I\nweight::F\n\n\n\n\n\n","category":"type"},{"location":"usage/#Main.FaceDetection.HaarLikeObject-Tuple{Tuple{Integer, Integer}, Tuple{Integer, Integer}, Integer, Integer, Integer, Integer}","page":"Usage","title":"Main.FaceDetection.HaarLikeObject","text":"HaarLikeObject(\n    feature_type::Tuple{Integer, Integer},\n    position::Tuple{Integer, Integer},\n    width::Integer,\n    height::Integer,\n    threshold::Integer,\n    polarity::Integer\n) -> HaarLikeObject\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.create_features-NTuple{6, Int64}","page":"Usage","title":"Main.FaceDetection.create_features","text":"create_features(\n    img_height::Int, img_width::Int,\n    min_feature_width::Int,\n    max_feature_width::Int,\n    min_feature_height::Int,\n    max_feature_height::Int\n) -> Array{HaarLikeObject, 1}\n\nIteratively creates the Haar-like feautures\n\nArguments\n\nimg_height::Integer: The height of the image\nimg_width::Integer: The width of the image\nmin_feature_width::Integer: The minimum width of the feature (used for computation efficiency purposes)\nmax_feature_width::Integer: The maximum width of the feature\nmin_feature_height::Integer: The minimum height of the feature\nmax_feature_height::Integer: The maximum height of the feature\n\nReturns\n\nfeatures::AbstractArray: an array of Haar-like features found for an image\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.determine_feature_size-Tuple{Vector{String}}","page":"Usage","title":"Main.FaceDetection.determine_feature_size","text":"determine_feature_size(\n    pictures::Vector{String}\n) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}\ndetermine_feature_size(\n    pos_training_path::String,\n    neg_training_path::String\n) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}\n\nTakes images and finds the best feature size for the image size.\n\nArguments\n\npictures::Vector{String}: a list of paths to the images\n\nOR\n\npos_training_path::String: the path to the positive training images\nneg_training_path::String: the path to the negative training images\n\nReturns\n\nmax_feature_width::Integer: the maximum width of the feature\nmax_feature_height::Integer: the maximum height of the feature\nmin_feature_height::Integer: the minimum height of the feature\nmin_feature_width::Integer: the minimum width of the feature\nmin_size_img::Tuple{Integer, Integer}: the minimum-sized image in the image directories\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.ensemble_vote-Union{Tuple{N}, Tuple{T}, Tuple{IntegralArray{T, N}, Vector{HaarLikeObject}}} where {T, N}","page":"Usage","title":"Main.FaceDetection.ensemble_vote","text":"ensemble_vote(int_img::IntegralArray, classifiers::AbstractArray) -> Integer\n\nClassifies given integral image (IntegralArray) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.\n\nThat is, the final strong classifier is\n\nh(x) = begincases\n1textif sum_t=1^Talpha_th_t(x)geqfrac12sum_t=1^Talpha_t\n0textotherwise\nendcases\ntext where alpha_t = logleft(frac1beta_tright)\n\nArguments\n\nint_img::IntegralArray{T, N}: Integral image to be classified\nclassifiers::Vector{HaarLikeObject}: List of classifiers\n\nReturns\n\nvote::Int8   1       ⟺ sum of classifier votes > 0   0       otherwise\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.ensemble_vote_all-Tuple{Vector{String}, Vector{HaarLikeObject}}","page":"Usage","title":"Main.FaceDetection.ensemble_vote_all","text":"ensemble_vote_all(images::Vector{String}, classifiers::Vector{HaarLikeObject}) -> Vector{Int8}\nensemble_vote_all(image_path::String, classifiers::Vector{HaarLikeObject})     -> Vector{Int8}\n\nGiven a path to images, loads images then classifies votes using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.\n\nArguments\n\nimages::Vector{String}: list of paths to images; OR image_path::String: Path to images dir\nclassifiers::Vector{HaarLikeObject}: List of classifiers\n\nReturns\n\nvotes::Vector{Int8}: A list of assigned votes (see ensemble_vote).\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.get_faceness-Union{Tuple{N}, Tuple{T}, Tuple{F}, Tuple{I}, Tuple{HaarLikeObject{I, F}, IntegralArray{T, N}}} where {I, F, T, N}","page":"Usage","title":"Main.FaceDetection.get_faceness","text":"get_faceness(feature::HaarLikeObject{I, F}, int_img::IntegralArray{T, N}) -> Number\n\nGet facelikeness for a given feature.\n\nArguments\n\nfeature::HaarLikeObject: given Haar-like feature (parameterised replacement of Python's self)\nint_img::IntegralArray: Integral image array\n\nReturns\n\nscore::Number: Score for given feature\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.get_score-Union{Tuple{N}, Tuple{T}, Tuple{F}, Tuple{I}, Tuple{HaarLikeObject{I, F}, IntegralArray{T, N}}} where {I, F, T, N}","page":"Usage","title":"Main.FaceDetection.get_score","text":"get_score(feature::HaarLikeObject, int_img::AbstractArray) -> Tuple{Number, Number}\n\nGet score for given integral image array.  This is the feature cascade.\n\nArguments\n\nfeature::HaarLikeObject: given Haar-like feature (parameterised replacement of Python's self)\nint_img::AbstractArray: Integral image array\n\nReturns\n\nscore::Number: Score for given feature\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.get_vote-Union{Tuple{N}, Tuple{T}, Tuple{F}, Tuple{I}, Tuple{HaarLikeObject{I, F}, IntegralArray{T, N}}} where {I, F, T, N}","page":"Usage","title":"Main.FaceDetection.get_vote","text":"get_vote(feature::HaarLikeObject, int_img::IntegralArray) -> Integer\n\nGet vote of this feature for given integral image.\n\nArguments\n\nfeature::HaarLikeObject: given Haar-like feature\nint_img::IntegralArray: Integral image array\n\nReturns\n\nvote::Integer:  1       ⟺ this feature votes positively  -1      otherwise\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.load_image-Tuple{String}","page":"Usage","title":"Main.FaceDetection.load_image","text":"load_image(image_path::String) -> Array{Float64, N}\n\nLoads an image as gray_scale\n\nArguments\n\nimage_path::String: Path to an image\n\nReturns\n\nIntegralArray{Float64, N}: An array of floating point values representing the image\n\n\n\n\n\n","category":"method"},{"location":"usage/#Main.FaceDetection.sum_region-Union{Tuple{N}, Tuple{T}, Tuple{IntegralArray{T, N}, CartesianIndex{N}, CartesianIndex{N}}} where {T, N}","page":"Usage","title":"Main.FaceDetection.sum_region","text":"sum_region(\n\tintegral_image_arr::AbstractArray,\n\ttop_left::Tuple{Int,Int},\n\tbottom_right::Tuple{Int,Int}\n) -> Number\n\nArguments\n\niA::IntegralArray{T, N}: The intermediate Integral Image\ntop_left::NTuple{N, Int}: coordinates of the rectangle's top left corner\nbottom_right::NTuple{N, Int}: coordinates of the rectangle's bottom right corner\n\nReturns\n\nsum::T The sum of all pixels in the given rectangle defined by the parameters top_left and bottom_right\n\n\n\n\n\n","category":"method"},{"location":"benchmarking/#Benchmarking-Results","page":"Benchmarking Results","title":"Benchmarking Results","text":"","category":"section"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"These are results from benchmarking the training process.  The following are benchmarking results from running equivalent programmes in both repositories.  These programmes uses ~10 thousand training images at 19 x 19 pixels each.","category":"page"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"Language of Implementation Commit Run Time in Seconds Number of Allocations Memory Usage\nPython 8772a28 480.0354 —ᵃ —ᵃ\nJulia 6fd8ca9e 19.9057 255600105 5.11 GiB","category":"page"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"ᵃI have not yet figured out benchmarking memory usage in Python.","category":"page"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"These results were run on this machine:","category":"page"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"julia> versioninfo()\nJulia Version 1.5.2\nCommit 539f3ce943 (2020-09-23 23:17 UTC)\nPlatform Info:\n  OS: macOS (x86_64-apple-darwin18.7.0)\n  CPU: Intel(R) Core(TM) i5-6360U CPU @ 2.00GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)","category":"page"},{"location":"benchmarking/#.6-Update","page":"Benchmarking Results","title":"1.6 Update","text":"","category":"section"},{"location":"benchmarking/","page":"Benchmarking Results","title":"Benchmarking Results","text":"A few months after the release of Julia 1.6, I did some performance considerations (there are already quite a few nice features that come with 1.6).  Now these are the benchmarking results (see benchmark/basic.jl) Language of Implementation | Commit | Run Time in Seconds | Number of Allocations | Memory Usage –- | –- | –- | –- | –- Julia | ??? | 8.165 | 249021919 | 5.01 GiB","category":"page"},{"location":"acknowledgements/#Acknowledgements","page":"A Few Acknowledgements","title":"Acknowledgements","text":"","category":"section"},{"location":"acknowledgements/","page":"A Few Acknowledgements","title":"A Few Acknowledgements","text":"Thank you to:","category":"page"},{"location":"acknowledgements/","page":"A Few Acknowledgements","title":"A Few Acknowledgements","text":"Simon Honberg for the original open-source Python code upon which this repository is largely based.  This has provided me with an easy-to-read and clear foundation for the Julia implementation of this algorithm;\nMichael Jones for (along with Tirta Susilo) suggesting the method for a facelike-ness measure;\nMahdi Rezaei for helping me understand the full process of Viola-Jones' object detection;\nYing Bi for always being happy to answer questions (which mainly turned out to be a lack of programming knowledge rather than conceptual; also with help from Bing Xue);\nMr. H. Lockwood and Mr. D. Peck are Comp. Sci. students who have answered a few questions of mine;\nFinally, the people in the Julia slack channel, for dealing with many (probably stupid) questions.  Just a few who come to mind: Micket, David Sanders, Eric Forgy, Jakob Nissen, and Roel.","category":"page"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"See also the examples directory in the repository for more examples.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using FaceDetection, Serialization # Serialization is so that you can save your results\n\nnum_classifiers = 10 # this is the number of Haar-like features you want to select\n\n# provide paths to directories of training images\npos_training_path = \"...\" # positive images are, for example, faces\nneg_training_path = \"...\" # negative images are, for example, non-faces.  However, the Viola-Jones algorithm is for object detection, not just for face detection\n\nmax_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = (1, 2, 3, 4) # or use the function to select reasonable sized feature parameters given your maximum image size (see below)\ndetermine_feature_size(pos_training_path, neg_training_path)\n\n# learn the features from\nclassifiers = learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)\n\n# provide paths to directories of testing images\npos_testing_path = \"...\"\nneg_testing_path = \"...\"\n\n# obtain results\nnum_faces, num_non_faces = length(filtered_ls(pos_testing_path)), length(filtered_ls(neg_testing_path));\ncorrect_faces = sum(ensemble_vote_all(pos_testing_path, classifiers));\ncorrect_non_faces = num_non_faces - sum(ensemble_vote_all(neg_testing_path, classifiers));\ncorrect_faces_percent = (correct_faces / num_faces) * 100;\ncorrect_non_faces_percent = (correct_non_faces / num_non_faces) * 100;\n\n# print results\nprintln(\"$(string(correct_faces, \"/\", num_faces)) ($(correct_faces_percent) %) of positive images were correctly identified.\")\nprintln(\"$(string(correct_non_faces, \"/\", num_non_faces)) ($(correct_non_faces_percent) %) of positive images were correctly identified.\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Alternatively, you can save the data stored by the training process and read from that data file:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using FaceDetection, Serialization # Serialization is so that you can save your results\n\nnum_classifiers = 10 # this is the number of Haar-like features you want to select\n\n# provide paths to directories of training images\npos_training_path = \"...\" # positive images are, for example, faces\nneg_training_path = \"...\" # negative images are, for example, non-faces.  However, the Viola-Jones algorithm is for object detection, not just for face detection\n\nmax_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = (1, 2, 3, 4) # or use the function to select reasonable sized feature parameters given your maximum image size (see below)\ndetermine_feature_size(pos_training_path, neg_training_path)\n\nvotes, features = get_feature_votes(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)\n\ndata_file = \"...\" # this is where you want to save your data\nserialize(data_file, (votes, features)); # write classifiers to file\n\nvotes, all_features = deserialize(data_file); # read from saved data\nclassifiers = learn(pos_training_path, neg_training_path, all_features, votes, num_classifiers)\n\n# provide paths to directories of testing images\npos_testing_path = \"...\"\nneg_testing_path = \"...\"\n\n# obtain results\nnum_faces, num_non_faces = length(filtered_ls(pos_testing_path)), length(filtered_ls(neg_testing_path));\ncorrect_faces = sum(ensemble_vote_all(pos_testing_path, classifiers));\ncorrect_non_faces = num_non_faces - sum(ensemble_vote_all(neg_testing_path, classifiers));\ncorrect_faces_percent = (correct_faces / num_faces) * 100;\ncorrect_non_faces_percent = (correct_non_faces / num_non_faces) * 100;\n\n# print results\nprintln(\"$(string(correct_faces, \"/\", num_faces)) ($(correct_faces_percent) %) of positive images were correctly identified.\")\nprintln(\"$(string(correct_non_faces, \"/\", num_non_faces)) ($(correct_non_faces_percent) %) of positive images were correctly identified.\")","category":"page"},{"location":"caveats/#Caveats","page":"Caveats","title":"Caveats","text":"","category":"section"},{"location":"caveats/","page":"Caveats","title":"Caveats","text":"See also issues for a list of some things yet to be implemented.","category":"page"},{"location":"caveats/","page":"Caveats","title":"Caveats","text":"Needs peer review for algorithmic correctness.\nIn the current implementation of the Viola-Jones algorithm, we have not implemented scaling features.  This means that you should ideally have your training set the same size as your test set.  To make this easier while we work on scaling features, we have implemented keyword arguments to the functions determine_feature_size and learn.  E.g.,","category":"page"},{"location":"caveats/","page":"Caveats","title":"Caveats","text":"julia> load_image(image_path, scale = true, scale_up = (200, 200))\n\njulia> determine_feature_size(pos_training_path, neg_training_path; scale = true, scale_to = (200, 200))\n\njulia> classifiers = learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = true, scale_to = (200, 200))\n\njulia> ensemble_vote_all(pos_testing_path, classifiers, scale = true, scale_to = (200, 200))","category":"page"},{"location":"#Home","page":"Home","title":"Home","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is a Julia implementation of Viola-Jones' Object Detection algorithm.  Although there is an OpenCV port in Julia, it seems to be ill-maintained.  As this algorithm was created for commercial use, there seem to be few widely-used or well-documented implementations of it on GitHub.  The implementation this repository is based off is Simon Hohberg's Pythonic repository, as it seems to be well written (and the most starred Python implementation on GitHub, though this is not necessarily a good measure). Julia and Python alike are easy to read and write in — my thinking was that this would be easy enough to replicate in Julia, except for Pythonic classes, where I would have to use structs (or at least easier to replicate from than, for example, C++ or JS — two other highly-starred repositories.).","category":"page"},{"location":"#Important-Note","page":"Home","title":"Important Note","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"I implore collaboration.  I am an undergraduate student with no formal education in computer science (or computer vision of any form for that matter); I am certain this code can be refined/optimised by better programmers than myself.  This package is still maturing, and as such there are some things I would still like to implement.  Please, help me out if you like!","category":"page"},{"location":"#How-it-works","page":"Home","title":"How it works","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In an over-simplified manner, the Viola-Jones algorithm has some four stages:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Takes an image, converts it into an array of intensity values (i.e., in grey-scale), and constructs an Integral Image, such that for every element in the array, the Integral Image element is the sum of all elements above and to the left of it.  This makes calculations easier for step 2.\nFinds Haar-like Features from Integral Image.\nThere is now a training phase using sets of faces and non-faces.  This phase uses something called Adaboost (short for Adaptive Boosting).  Boosting is one method of Ensemble Learning. There are other Ensemble Learning methods like Bagging, Stacking, &c.. The differences between Bagging, Boosting, Stacking are:\nBagging uses equal weight voting. Trains each model with a random drawn subset of training set.\nBoosting trains each new model instance to emphasize the training instances that previous models mis-classified. Has better accuracy comparing to bagging, but also tends to overfit.\nStacking trains a learning algorithm to combine the predictions of several other learning algorithms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Despite this method being developed at the start of the century, it is blazingly fast compared to some machine learning algorithms, and still widely used.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Finally, this algorithm uses Cascading Classifiers to identify faces.  (See page 12 of the original paper for the specific cascade).","category":"page"},{"location":"","page":"Home","title":"Home","text":"For a better explanation, read the paper from 2001, or see the Wikipedia page on this algorithm.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FaceDetection\nDocTestSetup = quote\n    using FaceDetection\nend","category":"page"},{"location":"#Adding-FaceDetection.jl","page":"Home","title":"Adding FaceDetection.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"FaceDetection\")","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}