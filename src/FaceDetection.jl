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
export displaymatrix, filtered_ls, load_image, ensemble_vote, 
    ensemble_vote_all, reconstruct, get_random_image, 
    generate_validation_image, get_faceness, determine_feature_size

ENV["FACE_DETECTION_DISPLAY_LOGGING"] = "true"
ENV["FACE_DETECTION_DISPLAY_WARN"]    = "true"

include("IntegralImage.jl")
include("HaarLikeFeature.jl")
include("Utils.jl") # Utils.jl exports HaarLikeFeature.jl functions
include("AdaBoost.jl")

end # end module
