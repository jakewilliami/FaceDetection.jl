#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

include("IntegralImage.jl")


# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = Array(FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR)

FeatureType = Dict{String, AbstractArray}("two_vertical" => (1, 2), "two_horizontal" => (2, 1), "three_horizontal" => (3,1), "three_vertical" => (1,3), "four" => (2, 2))
FeatureTypes = [FeatureType["two_vertical"], FeatureType["two_horizontal"], FeatureType["three_horizontal"], FeatureType["three_vertical"], FeatureType["four"]]

        
abstract type HaarType end


struct HaarLikeFeature <: HaarType
    feature_type::Any
    position::Array
    topLeft::Array
    bottomRight::Array
    width::Int64
    height::Int64
    threshold::Float64
    polarity::Int64
    weight::Float64
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeFeature(feature_type::Any, position::Array, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
    end # end constructor
end # end structure


#
function getScore(self::HaarLikeFeature, intImg::AbstractArray)
    """
    Get score for given integral image array.
    :param int_img: Integral image array
    :type int_img: numpy.ndarray
    :return: Score for given feature
    :rtype: float
    """
    # set instance of score outside of ifs
    score = 0
    
    if self.feature_type == FeatureType["two_vertical"]
        first = sumRegion(intImg, self.top_left, (self.top_left[1] + self.width, Int(self.top_left[2] + self.height / 2)))
        second = sumRegion(int_img, (self.top_left[1], Int(self.top_left[2] + self.height / 2)), self.bottom_right)
        score = first - second
    end
    return score
    
end # end getScore function



imgArr = getImageMatrix()
integralImageArr = toIntegralImage(imgArr)

# println(integralImageArr)

output = getScore(integralImageArr)

println(output)

# x = new HaarLikeFeature()

# function Base.show(io::IO, gs::HaarLikeFeature)
#     println(io, getScore(gs))
# end
#
# show(integralImageArr)

# def _get_feature_vote(feature, image):
#     return feature.get_vote(image)

# x.getScore(integralImageArr)

abstract type HaarFeatureType end

struct HaarLikeFeature <: HaarType
    position::Array{Int64, 2}
    topLeft::Array{Int64, 2}
    bottomRight::Array{Int64, 2}
    width::Int64
    height::Int64
    threshold::Float64
    polarity::Int64
    weight::Float64
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeFeature(position::Array, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
    end # end constructor
end # end structure


struct HaarFeatureTwoVertical <: HaarFeatureType end
struct HaarFeatureTwoHorizontal <: HaarFeatureType end

function score(::HaarFeatureTwoVertical, self::HaarLikeFeature)
    first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
    second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
    return first - second
end

function score(::HaarFeatureTwoHoritontal, self::HaarLikeFeature)
    first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
    second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
    third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
    return first - second + third
end
