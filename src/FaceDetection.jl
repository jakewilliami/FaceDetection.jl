#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
module FaceDetection

export toIntegralImage, sumRegion, FeatureTypes, HaarLikeObject, getScore, getVote, learn, _get_feature_vote, _create_features, displaymatrix, notifyUser, loadImages, getImageMatrix, ensembleVote, ensembleVoteAll, reconstruct, getRandomImage, generateValidationImage

include("IntegralImage.jl")
include("HaarLikeFeature.jl")
include("AdaBoost.jl")
include("Utils.jl")

using .IntegralImage: toIntegralImage, sumRegion
using .HaarLikeFeature: FeatureTypes, HaarLikeObject, getScore, getVote, getFacelikeness
using .AdaBoost: learn, _get_feature_vote, _create_features
using .Utils: displaymatrix, notifyUser, loadImages, getImageMatrix, ensembleVote, ensembleVoteAll, reconstruct, getRandomImage, generateValidationImage

end # end module
