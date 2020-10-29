#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    


const feature_types = (
    two_vertical = (1, 2), 
    two_horizontal = (2, 1),
    three_horizontal = (3, 1),
    three_vertical = (1, 3),
    four = (2, 2)
    )

abstract type HaarFeatureAbstractType end
# abstract type AbstractHaarLikeObject <: HaarFeatureAbstractType end

"""
    mutable struct HaarLikeObject{I<:Integer,F<:AbstractFloat}

        Struct representing a Haar-like feature.
"""
mutable struct HaarLikeObject{I<:Integer,F<:AbstractFloat} <: HaarFeatureAbstractType
    feature_type::Tuple{I, I}
    position::Tuple{I, I}
    top_left::Tuple{I, I}
    bottom_right::Tuple{I, I}
    width::I
    height::I
    threshold::I
    polarity::I
    weight::F
    #parametric struct to store the ints and floats efficiently


end # end structure

    # constructor; equivalent of __init__ method within class
    function HaarLikeObject(
        feature_type::Tuple{Integer, Integer},
        position::Tuple{Integer, Integer},
        width::Integer,
        height::Integer,
        threshold::Integer,
        polarity::Integer
    )

        #all this to make sure that everything is of se same size
        p1,p2 = position
        f1,f2 = feature_type
        p1,p2,f1,f2,width,height,threshold,polarity = promote(p1,p2,f1,f2,width,height,threshold,polarity)
        position = (p1,p2)
        feature_type = (f1,f2)
        top_left = position
        bottom_right = (position[1] + width, position[2] + height)
        weight = float(one(p1)) #to make a float of the same size
        

        HaarLikeObject(feature_type, position, top_left, bottom_right, width, height, threshold, polarity, weight)
    end # end constructor
"""
    get_score(feature::HaarLikeObject, int_img::Array) -> Tuple{Number, Number}
    
Get score for given integral image array.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array

# Returns

- `score::Number`: Score for given feature
"""
function get_score(feature::HaarLikeObject{I,F}, int_img::Array) where {I,F}
    score = zero(I)
    faceness = zero(I)
    _2f = F(2)
    _half = F(0.5)
    _third = F(1.0/3.0)
    _3f = F(3)
    if feature.feature_type == feature_types.two_vertical
        first = sum_region(int_img, feature.top_left, (feature.top_left[1] + feature.width, I(round(feature.top_left[2] + feature.height * _half))))
        second = sum_region(int_img, (feature.top_left[1], I(round(feature.top_left[2] + feature.height * _half))), feature.bottom_right)
        score = first - second
        faceness = I(1)
    elseif feature.feature_type == feature_types.two_horizontal
        first = sum_region(int_img, feature.top_left, (I(round(feature.top_left[1] + feature.width * _half)), feature.top_left[2] + feature.height))
        second = sum_region(int_img, (I(round(feature.top_left[1] + feature.width * _half)), feature.top_left[2]), feature.bottom_right)
        score = first - second
        faceness = I(2)
    elseif feature.feature_type == feature_types.three_horizontal
        first = sum_region(int_img, feature.top_left, (I(round(feature.top_left[1] + feature.width * _third)), feature.top_left[2] + feature.height))
        second = sum_region(int_img, (I(round(feature.top_left[1] + feature.width * _third)), feature.top_left[2]), (I(round(feature.top_left[1] + _2f *feature.width * _third)), feature.top_left[2] + feature.height))
        third = sum_region(int_img, (I(round(feature.top_left[1] + _2f *feature.width * _third)), feature.top_left[2]), feature.bottom_right)
        score = first - second + third
        faceness = I(3)
    elseif feature.feature_type == feature_types.three_vertical
        first = sum_region(int_img, feature.top_left, (feature.bottom_right[1], I(round(feature.top_left[2] + feature.height * _third))))
        second = sum_region(int_img, (feature.top_left[1], I(round(feature.top_left[2] + feature.height * _third))), (feature.bottom_right[1], I(round(feature.top_left[2] + _2f *feature.height * _third))))
        third = sum_region(int_img, (feature.top_left[1], I(round(feature.top_left[2] + _2f *feature.height * _third))), feature.bottom_right)
        score = first - second + third
        faceness  = I(4)
    elseif feature.feature_type == feature_types.four
        # top left area
        first = sum_region(int_img, feature.top_left, (I(round(feature.top_left[1] + feature.width * _half)), I(round(feature.top_left[2] + feature.height * _half))))
        # top right area
        second = sum_region(int_img, (I(round(feature.top_left[1] + feature.width * _half)), feature.top_left[2]), (feature.bottom_right[1], I(round(feature.top_left[2] + feature.height * _half))))
        # bottom left area
        third = sum_region(int_img, (feature.top_left[1], I(round(feature.top_left[2] + feature.height * _half))), (I(round(feature.top_left[1] + feature.width * _half)), feature.bottom_right[2]))
        # bottom right area
        fourth = sum_region(int_img, (I(round(feature.top_left[1] + feature.width * _half)), I(round(feature.top_left[2] + feature.height * _half))), feature.bottom_right)
        score = first - second - third + fourth
        faceness = I(5)
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
    score = first(get_score(feature, int_img)) # we only care about score here
    return (feature.weight * score) < (feature.polarity * feature.threshold) ? one(Int8) : -one(Int8)
end
