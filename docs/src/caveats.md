# Caveats

_See also [issues](https://github.com/jakewilliami/FaceDetection.jl/issues) for a list of some things yet to be implemented._

-  **Needs peer review for algorithmic correctness.**
- In the current implementation of the Viola-Jones algorithm, we have not implemented scaling features.  This means that you should ideally have your training set the same size as your test set.  To make this easier while we work on scaling features, we have implemented keyword arguments to the functions `determine_feature_size` and `learn`.  E.g.,
```julia
julia> load_image(image_path, scale = true, scale_up = (200, 200))

julia> determine_feature_size(pos_training_path, neg_training_path; scale = true, scale_to = (200, 200))

julia> classifiers = learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = true, scale_to = (200, 200))

julia> ensemble_vote_all(pos_testing_path, classifiers, scale = true, scale_to = (200, 200))
```
