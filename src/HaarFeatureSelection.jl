#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

include("IntegralImage.jl")


# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = Array(FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR)

FeatureType = Dict{String, Any}("two_vertical" => (1, 2), "two_horizontal" => (2, 1), "three_horizontal" => (3,1), "three_vertical" => (1,3), "four" => (2, 2))
FeatureTypes = [FeatureType["two_vertical"], FeatureType["two_horizontal"], FeatureType["three_horizontal"], FeatureType["three_vertical"], FeatureType["four"]]

        
abstract type HaarType end


struct HaarLikeFeature <: HaarType
    feature_type::Any
    position::Array{Int64, 2}
    topLeft::Array{Int64, 2}
    bottomRight::Array{Int64, 2}
    width::Int64
    height::Int64
    threshold::Float64
    polarity::Int64
    weight::Float64
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeFeature(feature_type::Any, position::Array{Int64, 2}, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
    end # end constructor
    
    #
    function getScore(self::HaarLikeFeature, intImg::AbstractArray)
        """
        Get score for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        score = 0
        
        if self.feature_type == FeatureType["two_vertical"]
            first = sumRegion(intImg, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sumRegion(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second
        end
        
    end # end getScore function
end # end structure



# @enum FeatureType begin
#     two_vert_a = 1
#     two_vert_b = 2
#     two_hor_a = 2
#     two_hor_b = 1
#     three_hor_a = 3
#     three_hor_b = 1
#     three_vert_a = 1
#     three_vert_b = 3
#     four_a = 2
#     four_b = 2
# end


# FeatureTypes = Array((FeatureType.two_vert_a, FeatureType.two_vert_b), (FeatureType.two_hor_a, FeatureType.two_hor_b), (FeatureType.three_hor_a, FeatureType.three_hor_b), (FeatureType.three_vert_a, FeatureType.three_vert_a), (FeatureType.four_a, FeatureType.four_b))










# def enum(**enums):
#     return type('Enum', (), enums)
#
# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]
#
#
# class HaarLikeFeature(object):
#     """
#     Class representing a haar-like feature.
#     """
#
#     def __init__(self, feature_type, position, width, height, threshold, polarity):
#         """
#         Creates a new haar-like feature.
#         :param feature_type: Type of new feature, see FeatureType enum
#         :type feature_type: violajonse.HaarLikeFeature.FeatureTypes
#         :param position: Top left corner where the feature begins (x, y)
#         :type position: (int, int)
#         :param width: Width of the feature
#         :type width: int
#         :param height: Height of the feature
#         :type height: int
#         :param threshold: Feature threshold
#         :type threshold: float
#         :param polarity: polarity of the feature -1 or 1
#         :type polarity: int
#         """
#         self.type = feature_type
#         self.top_left = position
#         self.bottom_right = (position[0] + width, position[1] + height)
#         self.width = width
#         self.height = height
#         self.threshold = threshold
#         self.polarity = polarity
#         self.weight = 1
#
#     def get_score(self, int_img):
#         """
#         Get score for given integral image array.
#         :param int_img: Integral image array
#         :type int_img: numpy.ndarray
#         :return: Score for given feature
#         :rtype: float
#         """
#         score = 0
#         if self.type == FeatureType.TWO_VERTICAL:
#             first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
#             second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
#             score = first - second
#         elif self.type == FeatureType.TWO_HORIZONTAL:
#             first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
#             second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)
#             score = first - second
#         elif self.type == FeatureType.THREE_HORIZONTAL:
#             first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
#             second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
#             third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
#             score = first - second + third
#         elif self.type == FeatureType.THREE_VERTICAL:
#             first = ii.sum_region(int_img, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
#             second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)), (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
#             third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)
#             score = first - second + third
#         elif self.type == FeatureType.FOUR:
#             # top left area
#             first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
#             # top right area
#             second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]), (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
#             # bottom left area
#             third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
#             # bottom right area
#             fourth = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)), self.bottom_right)
#             score = first - second - third + fourth
#         return score
#
#     def get_vote(self, int_img):
#         """
#         Get vote of this feature for given integral image.
#         :param int_img: Integral image array
#         :type int_img: numpy.ndarray
#         :return: 1 iff this feature votes positively, otherwise -1
#         :rtype: int
#         """
#         score = self.get_score(int_img)
#         return self.weight * (1 if score < self.polarity * self.threshold else -1)



imgArr = getImageMatrix()
integralImageArr = toIntegralImage(imgArr)

# println(integralImageArr)

x = new HaarLikeFeature()

def _get_feature_vote(feature, image):
    return feature.get_vote(image)

x.getScore(integralImageArr)
