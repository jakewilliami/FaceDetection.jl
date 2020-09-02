#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


module Utils

include("HaarLikeFeature.jl")
include("IntegralImage.jl")

using Images: save, load, Colors, clamp01nan, Gray, imresize
using ImageDraw: draw!, Polygon, Point
using .HaarLikeFeature: FeatureTypes, getVote, getScore ,HaarLikeObject
using .IntegralImage: toIntegralImage

export displaymatrix, notifyUser, loadImages, ensembleVoteAll, reconstruct, getRandomImage, generateValidationImage #, getImageMatrix, ensembleVote


function displaymatrix(M::AbstractArray)
    #=
    A function to show a big matrix on one console screen (similar to default `print` of numpy arrays in Python).
    
    parameter `M`: Some array [type: Abstract Array]
    
    return: A nice array to print [type: plain text
    =#
    return show(IOContext(stdout, :limit => true, :compact => true, :short => true), "text/plain", M)
end


function notifyUser(message::AbstractString)
    return println("\033[1;34m===>\033[0;38m\033[1;38m\t$message\033[0;38m")
end


function loadImages(imageDir::AbstractString)
    #=
    Given a path to a directory of images, recursively loads those
    
    parameter `imageDir`: path to a directory of images [type: Abstract String (path)]
    
    return `images`: a list of images from the path provided [type: Abstract Array]
    =#
        
    images = []
    
    for file in filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(imageDir, join=true, sort=false))
        images = push!(images, getImageMatrix(file))
    end
    
    return images
end


function getImageMatrix(imageFile::AbstractString)
    #=
    Takes an image and constructs a matrix of greyscale intensity values based on it.
    
    parameter `imageFile`: the path of the file of the image to be turned into an array [type: Abstract String (path)]
    
    return `imgArr`: The array of greyscale intensity values from the image [type: Absrtact Array]
    =#
    
    img = load(imageFile)
    imgArr = convert(Array{Float64}, Gray.(img))
    
    return imgArr
end
    
    
function ensembleVote(intImg::AbstractArray, classifiers::AbstractArray)
    #=
    Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.
    
    That is, the final strong classifier is $h(x)=\begin{cases}1&\text{if }\sum_{t=1}^{T}\alpha_th_t(x)\geq \frac{1}{2}\sum_{t=1}^{T}\alpha_t\\0&\text{otherwise}\end{cases}$, where $\alpha_t=\log{\left(\frac{1}{\beta_t}\right)}$
    
    parameter `intImg`: Integral image to be classified [type: AbstractArray]
    parameter `classifiers`: List of classifiers [type: AbstractArray (array of HaarLikeObjects)]

    return:
        1       ⟺ sum of classifier votes > 0
        0       otherwise
    [type: Integer]
    =#
    return sum([HaarLikeFeature.getVote(c, intImg) for c in classifiers]) >= 0 ? 1 : 0
end


function ensembleVoteAll(intImgs::AbstractArray, classifiers::AbstractArray)
    #=
    Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.
    
    parameter `intImg`: Integral image to be classified [type: AbstractArray]
    parameter `classifiers`: List of classifiers [type: AbstractArray (array of HaarLikeObjects)]

    return list of assigned labels:
        1       if image was classified positively
        0       otherwise
    [type: Abstract Arrays (array of Integers)]
    =#
    
    return Array(map(i -> ensembleVote(i, classifiers), intImgs))
end


function reconstruct(classifiers::AbstractArray, imgSize::Tuple)
    #=
    Creates an image by putting all given classifiers on top of each other producing an archetype of the learned class of object.
    
    parameter `classifiers`: List of classifiers [type: Abstract Array (array of HaarLikeObjects)]
    parameter `imgSize`: Tuple of width and height [Tuple]

    return `result`: Reconstructed image [type: PIL.Image??]
    =#
    
    image = zeros(imgSize)
    
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = ((1 + c.polarity)^2)/4
        if c.featureType == HaarLikeFeature.FeatureTypes[1] # two vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    image[c.topLeft[2] + y, c.topLeft[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[2] # two horizontal
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[3] # three horizontal
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[4] # three vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[5] # four
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/2))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    if iszero(mod(x, c.height/2))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        end
    end # end for c in classifiers
    # image .-= minimum(image) # equivalent to `min(image...)`
    # image ./= maximum(image)
    # image .*= 255
    #
    # image = replace!(image, NaN=>0.0) # change NaN to white (not that there should be any NaN values)
    #
    return image
end


function getRandomImage(facePath::AbstractString, nonFacePath::AbstractString="", nonFaces::Bool=false)
    #=
    Chooses a random image from a given two directories.
    
    parameter `facePath`: The path to the faces directory [type: Abstract String (path)]
    parameter `nonFacePath`: The path to the non-faces directory [type: Abstract String (path)]
    
    return `fileName`: The path to the file randomly chosen [type: Abstract String (path)]
    =#
    
    if nonFaces
        face = rand(Bool)
        fileName = rand(filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(face ? facePath : nonFacePath, join=true)))
        return fileName#, face
    else
        fileName = rand(filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(facePath, join=true)))
        return fileName#, face
    end
end


function scaleBox(topLeft::Tuple{Int64, Int64}, bottomRight::Tuple{Int64, Int64}, genisisSize::Tuple{Int64, Int64}, imgSize::Tuple{Int64, Int64})
    #=
    Scales the bounding box around classifiers if the image we are pasting it on is a different size to the original image.
    
    parameter `topLeft`: the top left of the Haar-like feature [type: Tuple]
    parameter `bottomRight`: the bottom right of the Haar-like feature [type: Tuple]
    parameter `genisisSize`: the size of the test images [type: Tuple]
    parameter `imgSize`: the size of the image which we are pasting the bounding box on top of [type: Tuple]
    
    return: the corners of the bounding box [type: Tuples]
    =#
    
    imageRatio = (imgSize[1]/genisisSize[1], imgSize[2]/genisisSize[2])
    
    bottomLeft = (topLeft[1], bottomRight[2])
    topRight = (bottomRight[1], topLeft[2])
    
    topLeft = convert.(Int, round.(topLeft .* imageRatio))
    bottomRight = convert.(Int, round.(bottomRight .* imageRatio))
    bottomLeft = convert.(Int, round.(bottomLeft .* imageRatio))
    topRight = convert.(Int, round.(topRight .* imageRatio))
    
    return topLeft, bottomLeft, bottomRight, topRight
end


function generateValidationImage(imagePath::AbstractString, classifiers::AbstractArray)
    #=
    ...?
    =#
    
    img = toIntegralImage(getImageMatrix(imagePath))
    imgSize = size(img)
    
    topLeft = (0,0)
    bottomRight = (0,0)
    bottomLeft = (0,0)
    topRight = (0,0)
    
    features = []
    
    for c in classifiers
        features = push!(features, (c.topLeft, c.bottomRight))
        # if c.featureType == HaarLikeFeature.FeatureTypes[1] # two vertical
        #     features = push!(features, (c.topLeft, c.bottomRight))
        # elseif c.featureType == HaarLikeFeature.FeatureTypes[2] # two horizontal
        # elseif c.featureType == HaarLikeFeature.FeatureTypes[3] # three horizontal
        # elseif c.featureType == HaarLikeFeature.FeatureTypes[4] # three vertical
        # elseif c.featureType == HaarLikeFeature.FeatureTypes[5] # four
        # end
    end
    
    # todo: figure out rectangles over boxes.  This includes finding the smallest image in size and converting the random input to that size if needed (requires importing another module).  bottom right needs to have smallest y but greatest x, etc.
    # todo: IF face is not a non-face, then draw box.  Don't force a box.
    
    chosenTopLeft = (0, 0)
    chosenBottomRight = (0, 0)
    
    for f in features # iterate through elements in features (e.g., Any[((9, 2), (17, 12)), ((11, 8), (19, 16))])
        # for t in f # iterate through inner tuples
            chosenTopLeft = chosenTopLeft < f[1] || chosenTopLeft < f[1] ? chosenTopLeft : f[1]
            chosenBottomRight = chosenBottomRight > f[2] || chosenBottomRight > f[2] ? chosenBottomRight : f[2]
        # end
    end
    
    # reasonableProportion = Int(round(0.26 * minimum(imgSize)))
    #
    # topLeft = (reasonableProportion, reasonableProportion)
    # bottomRight = (imgSize[1] - reasonableProportion, imgSize[2] - reasonableProportion)
    # bottomLeft = (reasonableProportion, imgSize[2] - reasonableProportion)
    # topRight = (imgSize[1] - reasonableProportion, reasonableProportion)
    
    topLeft = chosenTopLeft
    bottomRight = chosenBottomRight
    bottomLeft = (chosenTopLeft[1], chosenBottomRight[2])
    topRight = (chosenBottomRight[1], chosenTopLeft[2])
    
    boxDimensions = scaleBox(topLeft, bottomRight, (19, 19), imgSize)
    
    
    
    
    
    boxes = zeros(imgSize)
    
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = ((1 + c.polarity)^2)/4
        if c.featureType == HaarLikeFeature.FeatureTypes[1] # two vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.topLeft[2] + y, c.topLeft[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[2] # two horizontal
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    boxes[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[3] # three horizontal
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    boxes[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[4] # three vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == HaarLikeFeature.FeatureTypes[5] # four
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/2))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    if iszero(mod(x, c.height/2))
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        end
    end # end for c in classifiers
    
    # validationImage = load(imagePath) .+ boxes
    #
    # println(typeof(img))
    # println(typeof(boxes))
    # println(typeof(validationImage))
    
    
    
    # using TestImages, ImageDraw, ColorVectorSpace, ImageCore
    # img = testimage("lighthouse");
    
    # img = imresize(img, (19, 19))
    
    # [println(c.featureType) for c in classifiers]
    #
    for c in classifiers
        
        # boxDimensions = [c.topLeft, (c.topLeft[1], c.bottomRight[2]), c.bottomRight, (c.bottomRight[1], c.topLeft[2])]
        
        boxDimensions = scaleBox(c.topLeft, c.bottomRight, (19, 19), imgSize)
        
        save(joinpath(homedir(), "Desktop", "validation.png"), draw!(load(imagePath), Polygon([Point(boxDimensions[1]), Point(boxDimensions[2]), Point(boxDimensions[3]), Point(boxDimensions[4])])))
    end
    
    # box = Polygon([Point(boxDimensions[1]), Point(boxDimensions[2]), Point(boxDimensions[3]), Point(boxDimensions[4])])
    
    # return save(joinpath(homedir(), "Desktop", "validation.png"), draw!(load(imagePath), box))
    
#     #Arguments
# * `center::Point` : the center of the polygon
# * `side_count::Int` : number of sides of the polygon
# * `side_length::Real` : length of each side
# * `θ::Real` : orientation of the polygon w.r.t x-axis (in radians)

    
    # save("/Users/jakeireland/Desktop/test.png", Gray.(map(clamp01nan, newImg)))
end


end # end module
