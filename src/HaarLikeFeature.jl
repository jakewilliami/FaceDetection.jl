#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

include("IntegralImage.jl")

feature_types = Dict{String, Tuple{Integer, Integer}}("two_vertical" => (1, 2), "two_horizontal" => (2, 1), "three_horizontal" => (3, 1), "three_vertical" => (1, 3), "four" => (2, 2))

abstract type HaarFeatureAbstractType end
# abstract type AbstractHaarLikeObject <: HaarFeatureAbstractType end

#=
Struct representing a Haar-like feature.
=#
mutable struct HaarLikeObject <: HaarFeatureAbstractType
    feature_type::Tuple{Integer, Integer}
    position::Tuple{Integer, Integer}
    top_left::Tuple{Integer, Integer}
    bottom_right::Tuple{Integer, Integer}
    width::Integer
    height::Integer
    threshold::Integer
    polarity::Integer
    weight::AbstractFloat
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeObject(
        feature_type::Tuple{Integer, Integer},
        position::Tuple{Integer, Integer},
        width::Integer,
        height::Integer,
        threshold::Integer,
        polarity::Integer
    )
        top_left = position
        bottom_right = (position[1] + width, position[2] + height)
        weight = 1
        
        new(feature_type, position, top_left, bottom_right, width, height, threshold, polarity, weight)
    end # end constructor
end # end structure

#=
    get_score(feature::HaarLikeObject, int_img::Array) -> Tuple{Number, Number}
    
Get score for given integral image array.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array

# Returns

- `score::Number`: Score for given feature
=#
function get_score(feature::HaarLikeObject, int_img::Array)
    score = 0
    faceness = 0

    if feature.feature_type == feature_types["two_vertical"]
        first = sum_region(int_img, feature.top_left, (feature.top_left[1] + feature.width, Int(round(feature.top_left[2] + feature.height / 2))))
        second = sum_region(int_img, (feature.top_left[1], Int(round(feature.top_left[2] + feature.height / 2))), feature.bottom_right)
        score = first - second
        faceness = 1
    elseif feature.feature_type == feature_types["two_horizontal"]
        first = sum_region(int_img, feature.top_left, (Int(round(feature.top_left[1] + feature.width / 2)), feature.top_left[2] + feature.height))
        second = sum_region(int_img, (Int(round(feature.top_left[1] + feature.width / 2)), feature.top_left[2]), feature.bottom_right)
        score = first - second
        faceness = 2
    elseif feature.feature_type == feature_types["three_horizontal"]
        first = sum_region(int_img, feature.top_left, (Int(round(feature.top_left[1] + feature.width / 3)), feature.top_left[2] + feature.height))
        second = sum_region(int_img, (Int(round(feature.top_left[1] + feature.width / 3)), feature.top_left[2]), (Int(round(feature.top_left[1] + 2 * feature.width / 3)), feature.top_left[2] + feature.height))
        third = sum_region(int_img, (Int(round(feature.top_left[1] + 2 * feature.width / 3)), feature.top_left[2]), feature.bottom_right)
        score = first - second + third
        faceness = 3
    elseif feature.feature_type == feature_types["three_vertical"]
        first = sum_region(int_img, feature.top_left, (feature.bottom_right[1], Int(round(feature.top_left[2] + feature.height / 3))))
        second = sum_region(int_img, (feature.top_left[1], Int(round(feature.top_left[2] + feature.height / 3))), (feature.bottom_right[1], Int(round(feature.top_left[2] + 2 * feature.height / 3))))
        third = sum_region(int_img, (feature.top_left[1], Int(round(feature.top_left[2] + 2 * feature.height / 3))), feature.bottom_right)
        score = first - second + third
        faceness  = 4
    elseif feature.feature_type == feature_types["four"]
        # top left area
        first = sum_region(int_img, feature.top_left, (Int(round(feature.top_left[1] + feature.width / 2)), Int(round(feature.top_left[2] + feature.height / 2))))
        # top right area
        second = sum_region(int_img, (Int(round(feature.top_left[1] + feature.width / 2)), feature.top_left[2]), (feature.bottom_right[1], Int(round(feature.top_left[2] + feature.height / 2))))
        # bottom left area
        third = sum_region(int_img, (feature.top_left[1], Int(round(feature.top_left[2] + feature.height / 2))), (Int(round(feature.top_left[1] + feature.width / 2)), feature.bottom_right[2]))
        # bottom right area
        fourth = sum_region(int_img, (Int(round(feature.top_left[1] + feature.width / 2)), Int(round(feature.top_left[2] + feature.height / 2))), feature.bottom_right)
        score = first - second - third + fourth
        faceness = 5
    end
    
    return score, faceness
end

#=
    get_vote(feature::HaarLikeObject, int_img::AbstractArray) -> Integer

Get vote of this feature for given integral image.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array [type: Abstract Array]

# Returns

 - `vote::Integer`:
    1       ⟺ this feature votes positively
    -1      otherwise
=#
function get_vote(feature::HaarLikeObject, int_img::AbstractArray)
    score = get_score(feature, int_img)[1] # we only care about score here

    return (feature.weight * score) < (feature.polarity * feature.threshold) ? one(Int8) : -one(Int8)
end
