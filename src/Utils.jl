#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
using FileIO # for loading images
using QuartzImageIO, ImageMagick, ImageSegmentation, ImageFeatures # for reading images
using Colors # for making images greyscale
using Images # for channelview; converting images to matrices; for `reconstruct`
using ImageTransformations # for scaling high-quality images down


include("HaarLikeFeature.jl")
#=
adaboost imports haarlikefeature.jl

main (fda.jl) imports integralimage.jl, adaboost.jl, and utils.jl
=#


# For summing all numerical values in any nested array
deepsum(a::Number) = a
deepsum(a) = sum(deepsum.(a))

# function deepsum(X::Union{NTuple{N,T} where N, AbstractArray{T}, T}) where T<:Number
#     sum(X)
# end
# func
# tion deepsum(X::Union{Tuple, AbstractArray})
#     sum(X) do v
#         deepsum(v)
#     end
# end

deepdiv(a::Number, b::Number) = a / b
deepdiv(a,b) = deepdiv.(a, b)

deeptimes(a::Number, b::Number) = a * b
deeptimes(a,b) = deeptimes.(a, b)

deepfloat(a::Number) = a * 1.0
deepfloat(a) = deepfloat.(a)


# for adaboost
function partial(f,a...)
    #=
    Tentative partial function (like Python's partial function) which takes a function as input and *some of* its variables.
    
    parameter `f`: a function
    parameter `a`: one of the function's arguments
    
    for `...` syntax see https://en.wikibooks.org/wiki/Introducing_Julia/Functions#Functions_with_variable_number_of_arguments
    =#
        ( (b...) -> f(a...,b...) )
end


function loadImages(imageDir::AbstractString)
    #=
    Given a path to a directory of images, recursively loads those images.
    
    parameter `imageDir`: path to a directory of images [type: Abstract String (path)]
    
    return `images`: a list of images from the path provided [type: Abstract Array]
    =#
    
    # imageDir = joinpath(dirname(dirname(@__FILE__)), "test", "images")
    
    images = []
    
    for file in readdir(imageDir, join=true, sort=false) # join true for getImageMatrix to find file from absolute path
        if basename(file) == ".DS_Store" # silly macos stuff >:(
            continue
        end
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
    # println(eltype(img))
    # img = imresize(img, ratio=1/8)
    img = imresize(img, (10,10)) # for standardised size

    # imgArr = convert(Array{Float64}, channelview(img)) # for coloured images
    imgArr = convert(Array{Float64}, Colors.Gray.(img))
    # println(eltype(imgArr))
    
    # segments = ImageSegmentation.felzenszwalb(img, 100)
    # imgArr = ImageSegmentation.imshow(map(i->segment_mean(segments,i), labels_map(segments)))
    # imgArr = segment_labels(segments)
    
    return imgArr

    # print(img)
end

    
# to emulate python's function
function pow(x::Number, y::Number)
    return (x)^(y)
end
    
    
function ensembleVote(intImg::AbstractArray, classifiers::AbstractArray)
    #=
    Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.
    
    That is, the final strong classifier is $h(x)=\begin{cases}1&\text{if }\sum_{t=1}^{T}\alpha_th_t(x)\geq \frac{1}{2}\sum_{t=1}^{T}\alpha_t\\0&\text{otherwise}\end{cases}$, where $\alpha_t=\log{\left(\frac{1}{\beta_t}\right)}$
    
    parameter `intImg`: Integral image to be classified [type: AbstractArray]
    parameter `classifiers`: List of classifiers [type: AbstractArray (array of HaarLikeFeatures)]

    return:
        1       ⟺ sum of classifier votes > 0
        0       otherwise
    [type: Integer]
    =#
    
    # if sum([c.get_vote(int_img) for c in classifiers]) >= 0
    #     return 1
    # else
    #     return 0
    # end
    
    # println([typeof(c) for c in classifiers])
    # println(typeof(classifiers))
    return deepsum([getVote(c, intImg) for c in classifiers]) >= 0 ? 1 : 0
    
     # >= 0 ? 1 : 0
end

function ensembleVoteAll(intImgs::AbstractArray, classifiers::AbstractArray)
    #=
    Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.
    
    parameter `intImg`: Integral image to be classified [type: AbstractArray]
    parameter `classifiers`: List of classifiers [type: AbstractArray (array of HaarLikeFeatures)]

    return list of assigned labels:
        1       if image was classified positively
        0       otherwise
    [type: Abstract Arrays (array of Integers)]
    =#
    
    # [println(typeof(c)) for c in classifiers]
    # println(typeof(classifiers))
    
    # votePartial = [partial(ensembleVote, c) for c in classifiers]
    # votePartial = partial(ensembleVote, [c for c in classifiers])
    # votePartial = partial(ensembleVote, classifiers)
    #
    # return map(votePartial, intImgs)
    
    # map(i -> ensembleVote(classifiers, i), intImgs)
    return map(i -> ensembleVote(i, classifiers), intImgs)
    
    # votePartial = partial(ensembleVote, intImgs)
    #
    # return map(votePartial, classifiers)
    
    
    # map(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX] * featureWeight : weights[imgIDX] * featureWeight, 1:numImgs)
    #
    # map(i -> (), intImgs)
    #
    
end


function reconstruct(classifiers::AbstractArray, imgSize::Tuple)
    #=
    Creates an image by putting all given classifiers on top of each other producing an archetype of the learned class of object.
    
    parameter `classifiers`: List of classifiers [type: Abstract Array (array of HaarLikeFeatures)]
    parameter `imgSize`: Tuple of width and height [Tuple]

    return `result`: Reconstructed image [type: PIL.Image??]
    =#
    
    image = zeros(imgSize)
    
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.featureType == FeatureTypes[1] # two vertical
            # println("hi")
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                        # println(sign)
                    end
                    image[c.topLeft[2] + y, c.topLeft[1] + x] += 1 * sign * c.weight
                    # println(c.topLeft[2] + y, " , ", c.topLeft[1] + x)
                    # println(1*sign*c.weight)
                end
            end
        elseif c.featureType == FeatureTypes[2] # two horizontal
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.topLeft[0] + x, c.topLeft[1] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == FeatureTypes[3] # three horizontal
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == FeatureTypes[4] # three vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == FeatureTypes[5] # four
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
    # println(eltype(image))
    # image .-= minimum(image) # equivalent to `min(image...)`
    # image ./= maximum(image)
    # image .*= 255
    
    image = replace!(image, NaN=>0.0) # change NaN to white
    
    # println(eltype(image))
    
    
    # image -= image.min()
    # image /= image.max()
    # image -= min(image)
    # image /= max(image)
    # image *= 255 # for colours
    # result = Image.fromarray(image.astype(np.uint8)) # this is where the images come from.  Get these values from IntegralImage.getImageMatrix()
    
    # result = imshow(image, cmap="rainbow", fmt="png")
    # result = reinterpret(UInt8, image)
    # println(typeof(result))
    
    # return result
    # println(findmin(image))
    return image
end


function getRandomImage(facePath::AbstractString, nonFacePath::AbstractString)
    face = rand(Bool)
    fileName = rand(readdir(face ? facePath : nonFacePath, join=true))
    return fileName#, face
end


function generateValidationImage()
    images = map(load, [getRandomImage() for _ in 1:169])
    newImage = new("RGB", (256, 256))
    tlx = 0
    tly = 0
    
    for img in images
        new_img.paste(img, (tlx, tly))
        if tlx < 12*19
            tlx += 19
        else
            tlx = 0
            tly += 19
        end
    end
    
    save("/Users/jakeireland/Desktop/test.png", Gray.(map(clamp01nan, newImg)))
end



# export getImageMatrix
export loadImages
export ensembleVote
export ensembleVoteAll
export reconstruct
export partial
export deepsum
export deepfloat
export deepdiv
export deeptimes


### TESTING

# output = loadImages("/Users/jakeireland/FaceDetection.jl/test/images/")
#
# println(output)
