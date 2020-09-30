#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


include("HaarLikeFeature.jl")
include("IntegralImage.jl")

using Images: save, load, Colors, clamp01nan, Gray, imresize
using ImageDraw: draw!, Polygon, Point

#=
    displaymatrix(M::AbstractArray) -> AbstractString

A function to show a big matrix on one console screen (similar to default `print` of numpy arrays in Python).

# Arguments

- `M::AbstractArray`: Some array

# Returns
- `A::AbstractString`: A nice array to print
=#
function displaymatrix(M::AbstractArray)
    return show(IOContext(stdout, :limit => true, :compact => true, :short => true), "text/plain", M); print("\n")
end

#=
    notify_user(message::AbstractString) -> AbstractString

A function to pretty print a message to the user

# Arguments

- `message::AbstractString`: Some message to print

# Returns
- `A::AbstractString`: A message to print to the user
=#
function notify_user(message::AbstractString)
    return println("\033[1;34m===>\033[0;38m\033[1;38m\t$message\033[0;38m")
end

#=
    load_images(image_dir::AbstractString) -> Tuple{AbstractArray, AbstractArray}

Given a path to a directory of images, recursively loads those

# Arguments

- `image_dir::AbstractString`: path to a directory of images

# Returns

- `images::AbstractArray`: a list of images from the path provided
= `files::AbstractArray`: a list of file names of the images
=#
function load_images(image_dir::AbstractString)
    files = filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(image_dir, join=true, sort=false))
    images = []
    
    for file in files
        images = push!(images, get_image_matrix(file))
    end
    
    return images, files
end

#=
    get_image_matrix(image_file::AbstractString) -> AbstractArray
    
Takes an image and constructs a matrix of greyscale intensity values based on it.

# Arguments

- `image_file::AbstractString`: the path of the file of the image to be turned into an array

# Returns

- `img_arr::AbstractArray`: The array of greyscale intensity values from the image
=#
function get_image_matrix(image_file::AbstractString; scale_up::Bool=true)
    img = load(image_file)
    
    if scale_up
        img = imresize(img, (577, 577))
    end
    
    img_arr = convert(Array{Float64}, Gray.(img))
    
    return img_arr
end

#=
    determine_feature_size(
        pos_training_path::AbstractString,
        neg_training_path::AbstractString
    ) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}

Takes images and finds the best feature size for the image size.

# Arguments

- `pos_training_path::AbstractString`: the path to the positive training images
- `neg_training_path::AbstractString`: the path to the negative training images

# Returns

- `max_feature_width::Integer`: the maximum width of the feature
- `max_feature_height::Integer`: the maximum height of the feature
- `min_feature_height::Integer`: the minimum height of the feature
- `min_feature_width::Integer`: the minimum width of the feature
- `min_size_img::Tuple{Integer, Integer}`: the minimum-sized image in the image directories
=#
function determine_feature_size(
    pos_training_path::AbstractString,
    neg_training_path::AbstractString
)
    min_feature_height = 0
    min_feature_width = 0
    max_feature_height = 0
    max_feature_width = 0

    min_size_img = (0, 0)
    sizes = []

    for picture_dir in [pos_training_path, neg_training_path]
            for picture in filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(picture_dir, join=true, sort=false))
                new_size = size(load(joinpath(homedir(), "FaceDetection.jl", "data", picture_dir, picture)))
                sizes = push!(sizes, new_size)
            end
    end
    
    min_size_img = minimum(sizes)
    
    max_feature_height = Int(round(min_size_img[2]*(10/19)))
    max_feature_width = Int(round(min_size_img[1]*(10/19)))
    min_feature_height = Int(round(max_feature_height - max_feature_height*(2/max_feature_height)))
    min_feature_width = Int(round(max_feature_width - max_feature_width*(2/max_feature_width)))
    
    return max_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img
    
end

#=
    _ensemble_vote(int_img::AbstractArray, classifiers::AbstractArray) -> Integer

Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.

That is, the final strong classifier is $h(x)=\begin{cases}1&\text{if }\sum_{t=1}^{T}\alpha_th_t(x)\geq \frac{1}{2}\sum_{t=1}^{T}\alpha_t\\0&\text{otherwise}\end{cases}$, where $\alpha_t=\log{\left(\frac{1}{\beta_t}\right)}$

# Arguments

- `int_img::AbstractArray`: Integral image to be classified
- `classifiers::Array{HaarLikeObject, 1}`: List of classifiers

# Returns

- `vote::Integer`
    1       ⟺ sum of classifier votes > 0
    0       otherwise
=#
function _ensemble_vote(int_img::AbstractArray, classifiers::AbstractArray)
    # evidence = sum([max(get_vote(c[1], image), 0.) * c[2] for c in classifiers])
    # weightedSum = sum([c[2] for c in classifiers])
    # return evidence >= (weightedSum / 2) ? 1 : -1
    
    return sum([get_vote(c, int_img) for c in classifiers]) >= 0 ? 1 : 0
end

#=
    ensemble_vote_all(int_imgs::AbstractArray, classifiers::AbstractArray) -> AbstractArray
Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.

# Arguments
- `int_img::AbstractArray`: Integral image to be classified
- `classifiers::Array{HaarLikeObject, 1}`: List of classifiers

# Returns

`votes::AbstractArray`: A list of assigned votes (see _ensemble_vote).
=#
function ensemble_vote_all(int_imgs::AbstractArray, classifiers::AbstractArray)
    return Array(map(i -> _ensemble_vote(i, classifiers), int_imgs))
end

#=
    get_faceness(feature, int_img::AbstractArray) -> Number

Get facelikeness for a given feature.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array

# Returns

- `score::Number`: Score for given feature
=#
function get_faceness(feature, int_img::AbstractArray)
        score, faceness = get_score(feature, int_img)
        
        return (feature.weight * score) < (feature.polarity * feature.threshold) ? faceness : 0
end

#=
    reconstruct(classifiers::AbstractArray, img_size::Tuple) -> AbstractArray

Creates an image by putting all given classifiers on top of each other producing an archetype of the learned class of object.

# Arguments

- `classifiers::Array{HaarLikeObject, 1}`: List of classifiers
- `img_size::Tuple{Integer, Integer}`: Tuple of width and height

# Returns

- `result::AbstractArray`: Reconstructed image
=#
function reconstruct(classifiers::AbstractArray, img_size::Tuple)
    image = zeros(img_size)
    
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = ((1 + c.polarity)^2)/4
        if c.feature_type == feature_types["two_vertical"]
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    image[c.top_left[2] + y, c.top_left[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["two_horizontal"]
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["three_horizontal"]
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["three_vertical"]
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    image[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["four"]
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
    # image .-= minimum(image) # equivalent to `min(image...)`
    # image ./= maximum(image)
    # image .*= 255
    #
    # image = replace!(image, NaN=>0.0) # change NaN to white (not that there should be any NaN values)
    #
    return image
end

#=
    get_random_image(
        face_path::AbstractString,
        non_face_path::AbstractString="",
        non_faces::Bool=false
    ) -> AbstractString

Chooses a random image from a given two directories.

# Arguments

- `face_path::AbstractString`: The path to the faces directory
- `non_face_path::AbstractString`: The path to the non-faces directory

# Returns

- `file_name::AbstractString`: The path to the file randomly chosen
=#
function get_random_image(
    face_path::AbstractString,
    non_face_path::AbstractString="",
    non_faces::Bool=false
)
    file_name = ""
    
    if non_faces
        face = rand(Bool)
        file_name = rand(filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(face ? face_path : non_face_path, join=true)))
    else
        file_name = rand(filter!(f -> ! occursin(r".*\.DS_Store", f), readdir(face_path, join=true)))
    end
    
    return file_name
end

#=
    scale_box(
        top_left::Tuple{Integer, Integer},
        bottom_right::Tuple{Integer, Integer},
        genisis_size::Tuple{Integer, Integer},
        img_size::Tuple{Integer, Integer}
    ) -> NTuple{::Tuple{Integer, Integer}, 4}

Scales the bounding box around classifiers if the image we are pasting it on is a different size to the original image.

# Arguments

- `top_left::Tuple{Integer, Integer}`: the top left of the Haar-like feature
- `bottom_right::Tuple{Integer, Integer}`: the bottom right of the Haar-like feature
- `genisis_size::Tuple{Integer, Integer}`: the size of the test images
- `img_size::Tuple{Integer, Integer}`: the size of the image which we are pasting the bounding box on top of

# Returns

- `top_left::Tuple{Integer, Integer},`: new top left of box after scaling
- `bottom_left::Tuple{Integer, Integer},`: new bottom left of box after scaling
- `bottom_right::Tuple{Integer, Integer},`: new bottom right of box after scaling
- `top_right::Tuple{Integer, Integer},`: new top right of box after scaling
=#
function scale_box(
    top_left::Tuple{Integer, Integer},
    bottom_right::Tuple{Integer, Integer},
    genisis_size::Tuple{Integer, Integer},
    img_size::Tuple{Integer, Integer}
)
    image_ratio = (img_size[1]/genisis_size[1], img_size[2]/genisis_size[2])
    
    bottom_left = (top_left[1], bottom_right[2])
    top_right = (bottom_right[1], top_left[2])
    
    top_left = convert.(Int, round.(top_left .* image_ratio))
    bottom_right = convert.(Int, round.(bottom_right .* image_ratio))
    bottom_left = convert.(Int, round.(bottom_left .* image_ratio))
    top_right = convert.(Int, round.(top_right .* image_ratio))
    
    return top_left, bottom_left, bottom_right, top_right
end

#=
    generate_validation_image(image_path::AbstractString, classifiers::AbstractArray) -> AbstractArray
    
Generates a bounding box around the face of a random image.

# Arguments

- `image_path::AbstractString`: The path to images
- `classifiers::Array{HaarLikeObject, 1}`: List of classifiers/haar like features

# Returns

- `validation_image::AbstractArray`: The new image with a bounding box
=#
function generate_validation_image(image_path::AbstractString, classifiers::AbstractArray)
    img = to_integral_image(get_image_matrix(image_path))
    img_size = size(img)
    
    top_left = (0,0)
    bottom_right = (0,0)
    bottom_left = (0,0)
    top_right = (0,0)
    
    features = []
    
    for c in classifiers
        features = push!(features, (c.top_left, c.bottom_right))
        # if c.feature_type == feature_types[1] # two vertical
        #     features = push!(features, (c.top_left, c.bottom_right))
        # elseif c.feature_type == feature_types[2] # two horizontal
        # elseif c.feature_type == feature_types[3] # three horizontal
        # elseif c.feature_type == feature_types[4] # three vertical
        # elseif c.feature_type == feature_types[5] # four
        # end
    end
    
    # todo: figure out rectangles over boxes.  This includes finding the smallest image in size and converting the random input to that size if needed (requires importing another module).  bottom right needs to have smallest y but greatest x, etc.
    # todo: IF face is not a non-face, then draw box.  Don't force a box.
    
    chosen_top_left = (0, 0)
    chosen_bottom_right = (0, 0)
    
    for f in features # iterate through elements in features (e.g., Any[((9, 2), (17, 12)), ((11, 8), (19, 16))])
        # for t in f # iterate through inner tuples
            chosen_top_left = chosen_top_left < f[1] || chosen_top_left < f[1] ? chosen_top_left : f[1]
            chosen_bottom_right = chosen_bottom_right > f[2] || chosen_bottom_right > f[2] ? chosen_bottom_right : f[2]
        # end
    end
    
    # reasonableProportion = Int(round(0.26 * minimum(img_size)))
    #
    # top_left = (reasonableProportion, reasonableProportion)
    # bottom_right = (img_size[1] - reasonableProportion, img_size[2] - reasonableProportion)
    # bottom_left = (reasonableProportion, img_size[2] - reasonableProportion)
    # top_right = (img_size[1] - reasonableProportion, reasonableProportion)
    
    top_left = chosen_top_left
    bottom_right = chosen_bottom_right
    bottom_left = (chosen_top_left[1], chosen_bottom_right[2])
    top_right = (chosen_bottom_right[1], chosen_top_left[2])
    
    box_dimensions = scale_box(top_left, bottom_right, (19, 19), img_size)
    
    
    
    
    
    boxes = zeros(img_size)
    
    for c in classifiers
        # map polarity: -1 -> 0, 1 -> 1
        polarity = ((1 + c.polarity)^2)/4
        if c.feature_type == feature_types["two_vertical"]
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if y >= c.height/2
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.top_left[2] + y, c.top_left[1] + x] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["two_horizontal"]
            sign = polarity
            for x in 1:c.width
                if x >= c.width/2
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    boxes[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["three_horizontal"]
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/3))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    boxes[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["three_vertical"]
            for x in 1:c.width
                sign = polarity
                for y in 1:c.height
                    if iszero(mod(x, c.height/3))
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        elseif c.feature_type == feature_types["four"]
            sign = polarity
            for x in 1:c.width
                if iszero(mod(x, c.width/2))
                    sign = mod((sign + 1), 2)
                end
                for y in 1:c.height
                    if iszero(mod(x, c.height/2))
                        sign = mod((sign + 1), 2)
                    end
                    boxes[c.top_left[1] + x, c.top_left[2] + y] += 1 * sign * c.weight
                end
            end
        end
    end # end for c in classifiers
    
    # validationImage = load(image_path) .+ boxes
    #
    # println(typeof(img))
    # println(typeof(boxes))
    # println(typeof(validationImage))
    
    
    
    # using TestImages, ImageDraw, ColorVectorSpace, ImageCore
    # img = testimage("lighthouse");
    
    # img = imresize(img, (19, 19))
    
    # [println(c.feature_type) for c in classifiers]
    #
    for c in classifiers
        
        # box_dimensions = [c.top_left, (c.top_left[1], c.bottom_right[2]), c.bottom_right, (c.bottom_right[1], c.top_left[2])]
        
        box_dimensions = scale_box(c.top_left, c.bottom_right, (19, 19), img_size)
        
        save(joinpath(homedir(), "Desktop", "validation.png"), draw!(load(image_path), Polygon([Point(box_dimensions[1]), Point(box_dimensions[2]), Point(box_dimensions[3]), Point(box_dimensions[4])])))
    end
    
    
    # with open('classifiers_' + str(T) + '_' + hex(random.getrandbits(16)) + '.pckl', 'wb') as file:
    #     pickle.dump(classifiers, file)
    
    # box = Polygon([Point(box_dimensions[1]), Point(box_dimensions[2]), Point(box_dimensions[3]), Point(box_dimensions[4])])
    
    # return save(joinpath(homedir(), "Desktop", "validation.png"), draw!(load(image_path), box))
    
#     #Arguments
# * `center::Point` : the center of the polygon
# * `side_count::Int` : number of sides of the polygon
# * `side_length::Real` : length of each side
# * `θ::Real` : orientation of the polygon w.r.t x-axis (in radians)

    
    # save("/Users/jakeireland/Desktop/test.png", Gray.(map(clamp01nan, newImg)))
end
