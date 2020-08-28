#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


module Utils

include("HaarLikeFeature.jl")

using Images: save, load, Colors, clamp01nan, Gray
using .HaarLikeFeature: FeatureTypes, HaarLikeObject, getScore, getVote

export displaymatrix, notifyUser, loadImages, getImageMatrix, ensembleVote, ensembleVoteAll, reconstruct, getRandomImage, generateValidationImage


function displaymatrix(M::AbstractArray)
    #=
    A function to show a big matrix on one console screen (similar to default `print` of numpy arrays in Python).
    
    parameter `M`: Some array [type: Abstract Array]
    
    return: A nice array to print [type: plain text]
    =#
    return show(IOContext(stdout, :limit => true, :compact => true, :short => true), "text/plain", M)
end


function notifyUser(message::AbstractString)
    return println("\033[1;34m===>\033[0;38m\033[1;38m\t$message\033[0;38m")
end


function loadImages(imageDir::AbstractString)
    #=
    Given a path to a directory of images, recursively loads those images.
    
    parameter `imageDir`: path to a directory of images [type: Abstract String (path)]
    
    return `images`: a list of images from the path provided [type: Abstract Array]
    =#
        
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
    imgArr = convert(Array{Float64}, Colors.Gray.(img))
    
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
    return sum([getVote(c, intImg) for c in classifiers]) >= 0 ? 1 : 0
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
    
    # [println(typeof(c)) for c in classifiers]
    # println(typeof(classifiers))
    
    # votePartial = [partial(ensembleVote, c) for c in classifiers]
    # votePartial = partial(ensembleVote, [c for c in classifiers])
    # votePartial = partial(ensembleVote, classifiers)
    #
    # return map(votePartial, intImgs)
    
    # map(i -> ensembleVote(classifiers, i), intImgs)
    # return map(i -> ensembleVote(i, classifiers), intImgs)
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
        if c.featureType == FeatureTypes[1] # two vertical
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    image[c.topLeft[2] + y, c.topLeft[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.featureType == FeatureTypes[2] # two horizontal
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.topLeft[1] + x, c.topLeft[2] + y] += 1 * sign * c.weight
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
    # image .-= minimum(image) # equivalent to `min(image...)`
    # image ./= maximum(image)
    # image .*= 255
    #
    # image = replace!(image, NaN=>0.0) # change NaN to white (not that there should be any NaN values)
    #
    return image
end


function getRandomImage(facePath::AbstractString, nonFacePath::AbstractString)
    #=
    Chooses a random image from a given two directories.
    
    parameter `facePath`: The path to the faces directory [type: Abstract String (path)]
    parameter `nonFacePath`: The path to the non-faces directory [type: Abstract String (path)]
    
    return `fileName`: The path to the file randomly chosen [type: Abstract String (path)]
    =#
    
    face = rand(Bool)
    fileName = rand(readdir(face ? facePath : nonFacePath, join=true))
    return fileName
end


function generateValidationImage()
    #=
    ...?
    =#
    
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


end # end module
