#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
module FaceDetection

import Base: size, getindex, LinearIndices
using Images: Images, coords_spatial

export to_integral_image, sum_region
export learn
export feature_types, HaarLikeObject, get_score, get_vote
export displaymatrix, notify_user, filtered_ls, load_image,
    ensemble_vote_all, reconstruct, get_random_image,
    generate_validation_image, get_faceness, determine_feature_size


include("HaarLikeFeature.jl")
include("Utils.jl") # Utils.jl exports HaarLikeFeature.jl functions
include("IntegralImage.jl")
include("AdaBoost.jl")

end # end module
