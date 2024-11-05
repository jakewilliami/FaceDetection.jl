module FaceDetection

import Base: size, getindex, LinearIndices
using Images: Images, coords_spatial
using ProgressMeter: Progress, next!
using Base.Threads: @threads
using Base.Iterators: partition
using IntegralArrays # , IntervalSets

# export IntegralArray, to_integral_image, sum_region
export IntegralArray, sum_region
export learn, get_feature_votes
export FEATURE_TYPES, HaarLikeObject, get_score, get_vote
export displaymatrix,
    filtered_ls,
    load_image,
    ensemble_vote,
    ensemble_vote_all,
    reconstruct,
    get_random_image,
    generate_validation_image,
    get_faceness,
    determine_feature_size

# Setting these environment variables here doesn't do anything to the environment for some reason.
# Various functions in Adaboost will pull from the environment, defaulting to `"true"`.  If you
# need to turn off warnings and logging in the training phase, you can set these to `"false"` in
# your scripts.
# ENV["FACE_DETECTION_DISPLAY_LOGGING"] = "true"
# ENV["FACE_DETECTION_DISPLAY_WARN"]    = "true"

include("integral_image.jl")
include("haar_like_feature.jl")
include("utils.jl") # utils.jl exports haar_like_feature.jl functions
include("adaboost.jl")

end # end module
