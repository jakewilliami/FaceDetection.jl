#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

include("IntegralImage.jl")


# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = Array(FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR)

FeatureType = Dict{String, Tuple{Int64,Int64}}("two_vertical" => (1, 2), "two_horizontal" => (2, 1), "three_horizontal" => (3,1), "three_vertical" => (1,3), "four" => (2, 2))
FeatureTypes = [FeatureType["two_vertical"], FeatureType["two_horizontal"], FeatureType["three_horizontal"], FeatureType["three_vertical"], FeatureType["four"]]

        
abstract type HaarObject end
#
#
# struct HaarLikeFeature <: HaarType
#     feature_type::Any
#     position::Array
#     topLeft::Array
#     bottomRight::Array
#     width::Int64
#     height::Int64
#     threshold::Float64
#     polarity::Int64
#     weight::Float64
#
#     # constructor; equivalent of __init__ method within class
#     function HaarLikeFeature(feature_type::Any, position::Array, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
#         topLeft = position
#         bottomRight = (position[1] + width, position[2] + height)
#         weight = 1
#         new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
#     end # end constructor
# end # end structure
#
#
# #
# function getScore(self::HaarLikeFeature, intImg::AbstractArray)
#     """
#     Get score for given integral image array.
#     :param int_img: Integral image array
#     :type int_img: numpy.ndarray
#     :return: Score for given feature
#     :rtype: float
#     """
#     # set instance of score outside of ifs
#     score = 0
#
#     if self.feature_type == FeatureType["two_vertical"]
#         first = sumRegion(intImg, self.top_left, (self.top_left[1] + self.width, Int(self.top_left[2] + self.height / 2)))
#         second = sumRegion(int_img, (self.top_left[1], Int(self.top_left[2] + self.height / 2)), self.bottom_right)
#         score = first - second
#     end
#     return score
#
# end # end getScore function


abstract type HaarFeatureType end

# make structure
struct HaarLikeFeature <: HaarFeatureType#struct HaarLikeFeature{T} <: HaarObject where {T <: HaarFeatureType}
    position::Tuple{Int64, Int64}
    topLeft::Tuple{Int64, Int64}
    bottomRight::Tuple{Int64, Int64}
    width::Int64
    height::Int64
    threshold::Float64
    polarity::Int64
    weight::Float64
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeFeature(position::Tuple{Int64, Int64}, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
    end # end constructor
end # end structure


# define the various Haar like feature types
struct HaarFeatureTwoVertical <: HaarFeatureType end
struct HaarFeatureTwoHorizontal <: HaarFeatureType end
struct HaarFeatureThreeHorizontal <: HaarFeatureType end
struct HaarFeatureThreeVertical <: HaarFeatureType end
struct HaarFeatureFour <: HaarFeatureType end


# HaarFeatureTwoVertical = (1, 2)


# construct integral image
intImg = toIntegralImage(getImageMatrix())

function score(::HaarFeatureTwoVertical, self::HaarLikeFeature)
    first = sumRegion(intImg, self.top_left, (self.top_left[1] + self.width, Int(self.top_left[2] + self.height / 2)))
    second = sumRegion(intImg, (self.top_left[1], Int(self.top_left[2] + self.height / 2)), self.bottom_right)
    score = first - second
    return score
end

function score(::HaarFeatureTwoHorizontal, self::HaarLikeFeature)
    first = sumRegion(intImg, self.top_left, (int(self.top_left[1] + self.width / 3), self.top_left[2] + self.height))
    second = sumRegion(intImg, (Int(self.top_left[1] + self.width / 3), self.top_left[1]), (Int(self.top_left[1] + 2 * self.width / 3), self.top_left[2] + self.height))
    third = sumRegion(intImg, (Int(self.top_left[1] + 2 * self.width / 3), self.top_left[2]), self.bottom_right)
    score = first - second + third
    return score
end

function score(::HaarFeatureThreeHorizontal, self::HaarLikeFeature)
    first = sumRegion(intImg, self.top_left, (Int(self.top_left[1] + self.width / 3), self.top_left[2] + self.height))
    second = sumRegion(intImg, (Int(self.top_left[1] + self.width / 3), self.top_left[2]), (Int(self.top_left[1] + 2 * self.width / 3), self.top_left[2] + self.height))
    third = sumRegion(intImg, (Int(self.top_left[1] + 2 * self.width / 3), self.top_left[2]), self.bottom_right)
    score = first - second + third
    return score
end

function score(::HaarFeatureThreeVertical, self::HaarLikeFeature)
    first = sumRegion(intImg, self.top_left, (self.bottom_right[1], Int(self.top_left[2] + self.height / 3)))
    second = sumRegion(intImg, (self.top_left[1], Int(self.top_left[2] + self.height / 3)), (self.bottom_right[1], Int(self.top_left[2] + 2 * self.height / 3)))
    third = sumRegion(intImg, (self.top_left[1], Int(self.top_left[2] + 2 * self.height / 3)), self.bottom_right)
    score = first - second + third
    return score
end

function score(::HaarFeatureFour, self::HaarLikeFeature)
    # top left area
    first = sumRegion(intImg, self.top_left, (Int(self.top_left[1] + self.width / 2), Int(self.top_left[2] + self.height / 2)))
    # top right area
    second = sumRegion(intImg, (Int(self.top_left[1] + self.width / 2), self.top_left[2]), (self.bottom_right[1], Int(self.top_left[2] + self.height / 2)))
    # bottom left area
    third = sumRegion(intImg, (self.top_left[1], Int(self.top_left[2] + self.height / 2)), (Int(self.top_left[1] + self.width / 2), self.bottom_right[2]))
    # bottom right area
    fourth = sumRegion(intImg, (Int(self.top_left[1] + self.width / 2), int(self.top_left[2] + self.height / 2)), self.bottom_right)
    score = first - second - third + fourth
    return score
end



# imgArr = getImageMatrix()
# integralImageArr = toIntegralImage(imgArr)

# println(integralImageArr)

output = score(intImg)

println(output)
