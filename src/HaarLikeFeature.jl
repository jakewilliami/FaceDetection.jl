#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    

module HaarLikeFeature

include("IntegralImage.jl")

using .IntegralImage: toIntegralImage, sumRegion

export FeatureTypes, HaarLikeObject, getScore, getVote


FeatureTypes = [(1, 2), (2, 1), (3, 1), (1, 3), (2, 2)]


abstract type HaarFeatureAbstractType end
# abstract type AbstractHaarLikeObject <: HaarFeatureAbstractType end

# make structure
mutable struct HaarLikeObject <: HaarFeatureAbstractType
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
    weight::Float64
    
    # constructor; equivalent of __init__ method within class
    function HaarLikeObject(featureType::Tuple{Int64,Int64}, position::Tuple{Int64, Int64}, width::Int64, height::Int64, threshold::Int64, polarity::Int64)
        topLeft = position
        bottomRight = (position[1] + width, position[2] + height)
        weight = 1
        
        new(featureType, position, topLeft, bottomRight, width, height, threshold, polarity, weight)
    end # end constructor
end # end structure


function getScore(feature, intImg::Array)#function getScore(feature::HaarLikeObject, intImg::Array)
        #=
        Get score for given integral image array.
        
        parameter `feature`: given Haar-like feature (parameterised replacement of Python's `self`) [type: HaarLikeObject]
        parameter `intImg`: Integral image array [type: Abstract Array]
        
        return `score`: Score for given feature [type: Float]
        =#
        
        score = 0

        if feature.featureType == FeatureTypes[1] # two vertical
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
end



function getVote(feature, intImg::AbstractArray)#function getVote(feature::HaarLikeObject, intImg::AbstractArray)
    #=
    Get vote of this feature for given integral image.
    
    parameter `feature`: given Haar-like feature (parameterised replacement of Python's `self`) [type: HaarLikeObject]
    parameter `intImg`: Integral image array [type: Abstract Array]
    
    return:
        1       ⟺ this feature votes positively
        -1      otherwise
    [type: Integer]
    =#
    
    score = getScore(feature, intImg)
    
        
    return feature.weight * ((score < (feature.polarity * feature.threshold)) ? 1 : -1)
end


end # end module
