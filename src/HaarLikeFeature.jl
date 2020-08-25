#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

# include("IntegralImage.jl")
# include("Utils.jl")



# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = Array(FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR)

# FeatureType = Dict{String, Tuple{Int64,Int64}}("two_vertical" => (1, 2), "two_horizontal" => (2, 1), "three_horizontal" => (3,1), "three_vertical" => (1,3), "four" => (2, 2))
# FeatureTypes = [FeatureType["two_vertical"], FeatureType["two_horizontal"], FeatureType["three_horizontal"], FeatureType["three_vertical"], FeatureType["four"]]
FeatureTypes = [(1, 2), (2, 1), (3, 1), (1, 3), (2, 2)]

        
# abstract type HaarObject end
abstract type HaarFeatureType end

# make structure
mutable struct HaarLikeFeature <: HaarFeatureType#struct HaarLikeFeature{T} <: HaarObject where {T <: HaarFeatureType}
    #=
    Struct representing a Haar-like feature.
    =#
    
    featureType::Tuple{Int64, Int64}
    position::Tuple{Int64, Int64}
    topLeft::Tuple{Int64, Int64}
    bottomRight::Tuple{Int64, Int64}
    width::Int64
    height::Int64
    threshold::Int64
    polarity::Int64
    weight::Float64# number because when we update weight to featureWeight it changes to Float64
    # weight::Int64
    # weight::Any
    
    # constructor; equivalent of __init__ method within class # ::CartesianIndex
    function HaarLikeFeature(featureType::Tuple{Int64,Int64}, position::Tuple{Int64, Int64}, width::Int64, height::Int64, threshold::Int64, polarity::Int64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        
        new(featureType, position, topLeft, bottomRight, width, height, threshold, polarity)
    end # end constructor
end # end structure


# define the various Haar like feature types keeping the featureType parameter inside this object
# struct HaarFeatureTwoVertical <: HaarFeatureType end
# struct HaarFeatureTwoHorizontal <: HaarFeatureType end
# struct HaarFeatureThreeHorizontal <: HaarFeatureType end
# struct HaarFeatureThreeVertical <: HaarFeatureType end
# struct HaarFeatureFour <: HaarFeatureType end


# construct integral image
# intImg = toIntegralImage(getImageMatrix())

# if self.feature_type == FeatureType["two_vertical"] # like this. The value (1,2) or whatever it is, don't really matter.
# HAAR_FEATURETYPE_TWO_VERTICAL=(1, 2)

# function score(self::HaarLikeFeature)
#     # and dispatch on it here:
#     return score(self.feature_type, self)
# end
#
# function score(::HaarFeatureTwoVertical, self::HaarLikeFeature)
#     first = sumRegion(intImg, self.topLeft, (self.topLeft[1] + self.width, Int(self.topLeft[2] + self.height / 2)))
#     second = sumRegion(intImg, (self.topLeft[1], Int(self.topLeft[2] + self.height / 2)), self.bottomRight)
#     score = first - second
#     return score
# end
#
# function score(::HaarFeatureTwoHorizontal, self::HaarLikeFeature)
#     first = sumRegion(intImg, self.topLeft, (int(self.topLeft[1] + self.width / 3), self.topLeft[2] + self.height))
#     second = sumRegion(intImg, (Int(self.topLeft[1] + self.width / 3), self.topLeft[1]), (Int(self.topLeft[1] + 2 * self.width / 3), self.topLeft[2] + self.height))
#     third = sumRegion(intImg, (Int(self.topLeft[1] + 2 * self.width / 3), self.topLeft[2]), self.bottomRight)
#     score = first - second + third
#     return score
# end
#
# function score(::HaarFeatureThreeHorizontal, self::HaarLikeFeature)
#     first = sumRegion(intImg, self.topLeft, (Int(self.topLeft[1] + self.width / 3), self.topLeft[2] + self.height))
#     second = sumRegion(intImg, (Int(self.topLeft[1] + self.width / 3), self.topLeft[2]), (Int(self.topLeft[1] + 2 * self.width / 3), self.topLeft[2] + self.height))
#     third = sumRegion(intImg, (Int(self.topLeft[1] + 2 * self.width / 3), self.topLeft[2]), self.bottomRight)
#     score = first - second + third
#     return score
# end
#
# function score(::HaarFeatureThreeVertical, self::HaarLikeFeature)
#     first = sumRegion(intImg, self.topLeft, (self.bottomRight[1], Int(self.topLeft[2] + self.height / 3)))
#     second = sumRegion(intImg, (self.topLeft[1], Int(self.topLeft[2] + self.height / 3)), (self.bottomRight[1], Int(self.topLeft[2] + 2 * self.height / 3)))
#     third = sumRegion(intImg, (self.topLeft[1], Int(self.topLeft[2] + 2 * self.height / 3)), self.bottomRight)
#     score = first - second + third
#     return score
# end
#
# function score(::HaarFeatureFour, self::HaarLikeFeature)
#     # top left area
#     first = sumRegion(intImg, self.topLeft, (Int(self.topLeft[1] + self.width / 2), Int(self.topLeft[2] + self.height / 2)))
#     # top right area
#     second = sumRegion(intImg, (Int(self.topLeft[1] + self.width / 2), self.topLeft[2]), (self.bottomRight[1], Int(self.topLeft[2] + self.height / 2)))
#     # bottom left area
#     third = sumRegion(intImg, (self.topLeft[1], Int(self.topLeft[2] + self.height / 2)), (Int(self.topLeft[1] + self.width / 2), self.bottomRight[2]))
#     # bottom right area
#     fourth = sumRegion(intImg, (Int(self.topLeft[1] + self.width / 2), int(self.topLeft[2] + self.height / 2)), self.bottomRight)
#     score = first - second - third + fourth
#     return score
# end

function getScore(feature::HaarLikeFeature, intImg::Array)
        #=
        Get score for given integral image array.
        
        parameter `feature`: given Haar-like feature (parameterised replacement of Python's `self`) [type: HaarLikeFeature]
        parameter `intImg`: Integral image array [type: Abstract Array]
        
        return `score`: Score for given feature [type: Float]
        =#
        
        score = 0

        if feature.featureType == FeatureTypes[1] # two vertical
            # println((feature.topLeft[1] + feature.width, round(feature.topLeft[2] + feature.height / 2)))
            first = sumRegion(intImg, feature.topLeft, (feature.topLeft[1] + feature.width, Int(round(feature.topLeft[2] + feature.height / 2))))
            second = sumRegion(intImg, (feature.topLeft[1], Int(round(feature.topLeft[2] + feature.height / 2))), feature.bottomRight)
            score = first - second
        elseif feature.featureType == FeatureTypes[2] # two horizontal
            first = sumRegion(intImg, feature.topLeft, (Int(round(feature.topLeft[1] + feature.width / 2)), feature.topLeft[2] + feature.height))
            second = sumRegion(intImg, (Int(round(feature.topLeft[1] + feature.width / 2)), feature.topLeft[2]), feature.bottomRight)
            score = first - second
        elseif feature.featureType == FeatureTypes[3] # three horizontal
            first = sumRegion(intImg, feature.topLeft, (Int(round(feature.topLeft[1] + feature.width / 3)), feature.topLeft[2] + feature.height))
            second = sumRegion(intImg, (Int(round(feature.topLeft[1] + feature.width / 3)), feature.topLeft[2]), (Int(round(feature.topLeft[1] + 2 * feature.width / 3)), feature.topLeft[2] + feature.height))
            third = sumRegion(intImg, (Int(round(feature.topLeft[1] + 2 * feature.width / 3)), feature.topLeft[2]), feature.bottomRight)
            score = first - second + third
        elseif feature.featureType == FeatureTypes[4] # three vertical
            first = sumRegion(intImg, feature.topLeft, (feature.bottomRight[1], Int(round(feature.topLeft[2] + feature.height / 3))))
            second = sumRegion(intImg, (feature.topLeft[1], Int(round(feature.topLeft[2] + feature.height / 3))), (feature.bottomRight[1], Int(round(feature.topLeft[2] + 2 * feature.height / 3))))
            third = sumRegion(intImg, (feature.topLeft[1], Int(round(feature.topLeft[2] + 2 * feature.height / 3))), feature.bottomRight)
            score = first - second + third
        elseif feature.featureType == FeatureTypes[5] # four
            # top left area
            first = sumRegion(intImg, feature.topLeft, (Int(round(feature.topLeft[1] + feature.width / 2)), Int(round(feature.topLeft[2] + feature.height / 2))))
            # top right area
            second = sumRegion(intImg, (Int(round(feature.topLeft[1] + feature.width / 2)), feature.topLeft[2]), (feature.bottomRight[1], Int(round(feature.topLeft[2] + feature.height / 2))))
            # bottom left area
            third = sumRegion(intImg, (feature.topLeft[1], Int(round(feature.topLeft[2] + feature.height / 2))), (Int(round(feature.topLeft[1] + feature.width / 2)), feature.bottomRight[2]))
            # bottom right area
            fourth = sumRegion(intImg, (Int(round(feature.topLeft[1] + feature.width / 2)), Int(round(feature.topLeft[2] + feature.height / 2))), feature.bottomRight)
            score = first - second - third + fourth
        end
        
        return score
        
        
        # if feature.featureType == FeatureTypes[1] # two vertical
        #     first = sumRegion(intImg, feature.topLeft, (feature.topLeft[1] + feature.width, round(feature.topLeft[2] + feature.height / 2)))
        #     second = sumRegion(intImg, (feature.topLeft[1], round(feature.topLeft[2] + feature.height / 2)), feature.bottomRight)
        #     score = first - second
        # elseif feature.featureType == FeatureTypes[2] # two horizontal
        #     first = sumRegion(intImg, feature.topLeft, (round(feature.topLeft[1] + feature.width / 2), feature.topLeft[2] + feature.height))
        #     second = sumRegion(intImg, (round(feature.topLeft[1] + feature.width / 2), feature.topLeft[2]), feature.bottomRight)
        #     score = first - second
        # elseif feature.featureType == FeatureTypes[3] # three horizontal
        #     first = sumRegion(intImg, feature.topLeft, (round(feature.topLeft[1] + feature.width / 3), feature.topLeft[2] + feature.height))
        #     second = sumRegion(intImg, (round(feature.topLeft[1] + feature.width / 3), feature.topLeft[2]), (round(feature.topLeft[1] + 2 * feature.width / 3), feature.topLeft[2] + feature.height))
        #     third = sumRegion(intImg, (round(feature.topLeft[1] + 2 * feature.width / 3), feature.topLeft[2]), feature.bottomRight)
        #     score = first - second + third
        # elseif feature.featureType == FeatureTypes[4] # three vertical
        #     first = sumRegion(intImg, feature.topLeft, (feature.bottomRight[1], round(feature.topLeft[2] + feature.height / 3)))
        #     second = sumRegion(intImg, (feature.topLeft[1], round(feature.topLeft[2] + feature.height / 3)), (feature.bottomRight[1], round(feature.topLeft[2] + 2 * feature.height / 3)))
        #     third = sumRegion(intImg, (feature.topLeft[1], round(feature.topLeft[2] + 2 * feature.height / 3)), feature.bottomRight)
        #     score = first - second + third
        # elseif feature.featureType == FeatureTypes[5] # four
        #     # top left area
        #     first = sumRegion(intImg, feature.topLeft, (round(feature.topLeft[1] + feature.width / 2), round(feature.topLeft[2] + feature.height / 2)))
        #     # top right area
        #     second = sumRegion(intImg, (round(feature.topLeft[1] + feature.width / 2), feature.topLeft[2]), (feature.bottomRight[1], round(feature.topLeft[2] + feature.height / 2)))
        #     # bottom left area
        #     third = sumRegion(intImg, (feature.topLeft[1], round(feature.topLeft[2] + feature.height / 2)), (round(feature.topLeft[1] + feature.width / 2), feature.bottomRight[2]))
        #     # bottom right area
        #     fourth = sumRegion(intImg, (round(feature.topLeft[1] + feature.width / 2), round(feature.topLeft[2] + feature.height / 2)), feature.bottomRight)
        #     score = first - second - third + fourth
        # end
        #
        # return score
end



function getVote(feature::HaarLikeFeature, intImg::AbstractArray)
    #=
    Get vote of this feature for given integral image.
    
    parameter `feature`: given Haar-like feature (parameterised replacement of Python's `self`) [type: HaarLikeFeature]
    parameter `intImg`: Integral image array [type: Abstract Array]
    
    return:
        1       ⟺ this feature votes positively
        -1      otherwise
    [type: Integer]
    =#
    
    score = getScore(feature, intImg)
    
    
    # return feature.weight * (1 if score < feature.polarity * feature.threshold else -1)
    
    return feature.weight * ((score < (feature.polarity * feature.threshold)) ? 1 : -1)
    # return 1 * ((score < (feature.polarity * feature.threshold)) ? 1 : -1)
end


export getScore
export getVote


##### TESTING

# imgArr = getImageMatrix()
# integralImageArr = toIntegralImage(imgArr)

# println(integralImageArr)

# output = score(intImg)
#
# println(output)










# struct HaarLikeFeature <: HaarType
#     position::Array{Int64, 2}
#     topLeft::Array{Int64, 2}
#     bottomRight::Array{Int64, 2}
#     width::Int64
#     height::Int64
#     threshold::Float64
#     polarity::Int64
#     weight::Float64
#     # constructor; equivalent of __init__ method within class
#     function HaarLikeFeature(position::Array{Int64, 2}, width::Int64, height::Int64, threshold::Float64, polarity::Int64, weight::Float64)
#         topLeft = position
#         bottomRight = (position[1] + width, position[2] + height)
#         weight = 1
#         new(feature_type, topLeft, bottomRight, width, height, threshold, polarity)
#     end # end constructor
# abstract type HaarFeatureType end
# struct HaarFeatureTwoVertical <: HaarFeatureType end
# struct HaarFeatureTwoHorizontal <: HaarFeatureType end
# function score(::HaarFeatureTwoVertical, self::HaarLikeFeature)
#     first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
#     second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
#     return first - second
# end
# function score(::HaarFeatureTwoHoritontal, self::HaarLikeFeature)
#     first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
#     second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
#     third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
#     return first - second + third
# end
#
#
# struct HaarLikeFeature <: HaarType
#     # keep the feature_type parameter inside this object
#     feature_type::Union{HaarFeatureTwoVertical, HaarFeatureTwoHorizontal, etc.}
#     position::Array{Int64, 2}
#     ...
# end
# function score(self::HaarLikeFeature)
#     # and dispatch on it here:
#     return score(self.feature_type, self)
# end
