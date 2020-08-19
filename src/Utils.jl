#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
using FileIO # for loading images
using QuartzImageIO, ImageMagick, ImageSegmentation, ImageFeatures # for reading images
using Colors # for making images greyscale
using Images # for channelview; converting images to matrices
using ImageTransformations # for scaling high-quality images down


include("HaarLikeFeature.jl")
#=
adaboost imports haarlikefeature.jl

main (fda.jl) imports integralimage.jl, adaboost.jl, and utils.jl
=#


# for adaboost
function partial(f,a...) # for ... syntax see https://en.wikibooks.org/wiki/Introducing_Julia/Functions#Functions_with_variable_number_of_arguments
        ( (b...) -> f(a...,b...) )
end


function getImageMatrix(imageFile::AbstractString)
    img = load(imageFile)
    # img = imresize(img, ratio=1/8)
    img = imresize(img, (10,10)) # for standardised size

    # imgArr = convert(Array{Float64}, channelview(img)) # for coloured images
    imgArr = convert(Array{Float64}, Colors.Gray.(img))
    
    # segments = ImageSegmentation.felzenszwalb(img, 100)
    # imgArr = ImageSegmentation.imshow(map(i->segment_mean(segments,i), labels_map(segments)))
    # imgArr = segment_labels(segments)
    
    return imgArr

    # print(img)
end


function loadImages(imageDir::AbstractString)
    # imageDir = joinpath(dirname(dirname(@__FILE__)), "test", "images")
    
    images = []
    
    for file in readdir(imageDir, join=true, sort=false) # join true for getImageMatrix to find file
        if basename(file) == ".DS_Store" # silly macos stuff >:(
            continue
        end
        images = push!(images, getImageMatrix(file))
    end
    
    return images
end

    
# to emulate python's function
function pow(x::Number, y::Number)
    return (x)^(y)
end
    
    
function ensembleVote(intImg::AbstractArray, classifiers::AbstractArray)
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    if sum([c.get_vote(int_img) for c in classifiers]) >= 0
        return 1
    else
        return 0
    end
end

function ensembleVoteAll(intImgs::AbstractArray, classifiers::AbstractArray)
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    partial = (f, a...) -> (b...) -> f(a..., b...)
    votePartial = partial(ensembleVote, classifiers=classifiers)
    return map(votePartial, intImgs)
end


function reconstruct(classifiers::HaarLikeFeature, imgSize::Tuple)
    """
    Creates an image by putting all given classifiers on top of each other
    producing an archetype of the learned class of object.
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :param img_size: Tuple of width and height
    :type img_size: (int, int)
    :return: Reconstructed image
    :rtype: PIL.Image
    """
    image = zeros(imgSize)
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    image[c.top_left[2] + y, c.top_left[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.type == FeatureType.TWO_HORIZONTAL
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
                end
            end
        elseif c.type == FeatureType.THREE_HORIZONTAL
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.type == FeatureType.THREE_VERTICAL
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.type == FeatureType.FOUR
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/2))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    if iszero(mod(x, c.height/2))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        end
    end # end for c in classifiers
    
    image -= image.min()
    image /= image.max()
    image *= 255 # for colours
    # result = Image.fromarray(image.astype(np.uint8)) # this is where the images come from.  Get these values from IntegralImage.getImageMatrix()
    
    return result
end



# export getImageMatrix
export loadImages
export ensembleVote
export ensembleVoteAll
export reconstruct


### TESTING

# output = loadImages("/Users/jakeireland/FaceDetection.jl/test/images/")
#
# println(output)
