#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#



using Images: save, load, Colors, clamp01nan, Gray, imresize
using ImageDraw: draw, Polygon, Point

#=
    notify_user(message::String) -> String

A function to pretty print a message to the user

# Arguments

- `message::String`: Some message to print

# Returns
- `A::String`: A message to print to the user
=#
function notify_user(io::IO, message::String)
    return println(io, "\033[1;34m===>\033[0;38m\033[1;38m\t", message, "\033[0;38m")
end
notify_user(msg::String) = notify_user(stdout, msg)

#=
    filtered_ls(path::String) -> Vector{String}
    
A function to filter the output of readdir

# Arguments

- `path::AbstractString`: Some path to folder

# Returns

- `Vector{String}`: An array of filtered files in the path
=#
function filtered_ls(path::String)
    return filter!(f -> !occursin(r".*\.DS_Store", f), readdir(path, join=true, sort=false))
end

"""
    load_image(image_path::String) -> Array{Float64, N}
Loads an image as gray_scale

# Arguments
- `image_path::String`: Path to an image

# Returns

 - `Array{Float64, N}`: An array of floating point values representing the image
"""
function load_image(
    image_path::String;
    scale::Bool=false,
    scale_to::Tuple=(200,200)
    )

    img = load(image_path)
    img = convert(Array{Float64}, Gray.(img))
    img = scale ? imresize(img, scale_to) : img
    
    return to_integral_image(img)
end

"""
    determine_feature_size(
        pos_training_path::String,
        neg_training_path::String
    ) -> Tuple{Integer, Integer, Integer, Integer, Tuple{Integer, Integer}}

Takes images and finds the best feature size for the image size.

# Arguments

- `pos_training_path::String`: the path to the positive training images
- `neg_training_path::String`: the path to the negative training images

# Returns

- `max_feature_width::Integer`: the maximum width of the feature
- `max_feature_height::Integer`: the maximum height of the feature
- `min_feature_height::Integer`: the minimum height of the feature
- `min_feature_width::Integer`: the minimum width of the feature
- `min_size_img::Tuple{Integer, Integer}`: the minimum-sized image in the image directories
"""
function determine_feature_size(
    pos_training_path::String,
    neg_training_path::String;
    scale::Bool=false,
    scale_to::Tuple=(200, 200)
)

    min_feature_height = 0
    min_feature_width = 0
    max_feature_height = 0
    max_feature_width = 0

    min_size_img = (0, 0)
    sizes = Tuple{Int, Int}[]

    for picture_dir in[pos_training_path, neg_training_path]
        for picture in filtered_ls(picture_dir)
            img = load_image(picture, scale=scale, scale_to=scale_to)
            new_size = size(img)
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

@doc raw"""
    ensemble_vote(int_img::AbstractArray, classifiers::AbstractArray) -> Integer

Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.

That is, the final strong classifier is

```math
h(x) = \begin{cases}
1&\text{if }\sum_{t=1}^{T}\alpha_{th_{t(x)}}\geq\frac{1}{2}\sum_{t=1}^{T}\alpha_t\\
0&\text{otherwise}
\end{cases}
\text{ where }\alpha_t = \log{\left(\frac{1}{\beta_t}\right)}
```

# Arguments

- `int_img::Array{T, N}`: Integral image to be classified
- `classifiers::Vector{HaarLikeObject}`: List of classifiers

# Returns

- `vote::Int8`
    1       ⟺ sum of classifier votes > 0
    0       otherwise
"""
function ensemble_vote(int_img::Array{T, N}, classifiers::Vector{HaarLikeObject}) where {T, N}
    return sum(get_vote(c, int_img) for c in classifiers) ≥ zero(Int8) ? one(Int8) : zero(Int8)
    # return sum(c -> get_vote(c, int_img), classifiers) ≥ zero(Int8) ? one(Int8) : zero(Int8)
end

"""
    ensemble_vote_all(int_imgs::AbstractArray, classifiers::AbstractArray) -> Vector{Int8}
Classifies given integral image (Abstract Array) using given classifiers.  I.e., if the sum of all classifier votes is greater 0, the image is classified positively (1); else it is classified negatively (0). The threshold is 0, because votes can be +1 or -1.

# Arguments
- `int_img::AbstractArray`: Integral image to be classified
- `classifiers::Vector{HaarLikeObject}`: List of classifiers

# Returns

`votes::AbstractArray`: A list of assigned votes (see ensemble_vote).
"""
function ensemble_vote_all(
    image_path::String,
    classifiers::Vector{HaarLikeObject};
    scale::Bool=false,
    scale_to::Tuple=(200, 200)
    )
    
    return Int8[ensemble_vote(load_image(i, scale=scale, scale_to=scale_to), classifiers) for i in filtered_ls(image_path)]
end

"""
    get_faceness(feature::HaarLikeObject{I, F}, int_img::Array{T, N}) -> Number

Get facelikeness for a given feature.

# Arguments

- `feature::HaarLikeObject`: given Haar-like feature (parameterised replacement of Python's `self`)
- `int_img::AbstractArray`: Integral image array

# Returns

- `score::Number`: Score for given feature
"""
function get_faceness(feature::HaarLikeObject{I, F}, int_img::Array{T, N}) where {I, F, T, N}
    score, faceness = get_score(feature, int_img)
    
    return (feature.weight * score) < (feature.polarity * feature.threshold) ? faceness : zero(T)
end

#=
    reconstruct(classifiers::Vector, img_size::Tuple) -> AbstractArray

Creates an image by putting all given classifiers on top of each other producing an archetype of the learned class of object.

# Arguments

- `classifiers::Vector{HaarLikeObject}`: List of classifiers
- `img_size::Tuple{Integer, Integer}`: Tuple of width and height

# Returns

- `result::AbstractArray`: Reconstructed image
=#
function reconstruct(classifiers::Vector{HaarLikeObject{I, F}}, img_size::Tuple{Int, Int}) where {I, F}
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
        face_path::String,
        non_face_path::String="",
        non_faces::Bool=false
    ) -> AbstractString

Chooses a random image from a given two directories.

# Arguments

- `face_path::String`: The path to the faces directory
- `non_face_path::String`: The path to the non-faces directory

# Returns

- `file_name::AbstractString`: The path to the file randomly chosen

TODO: change this to check if `isempty(non_face_path)` instead of having another parameter
=#
function get_random_image(
    face_path::String;
    non_face_path::String="",
    non_faces::Bool=false
)
    file_name = string()
    
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
    T = typeof(first(top_left))
    image_ratio = (img_size[1]/genisis_size[1], img_size[2]/genisis_size[2])
    
    bottom_left = (top_left[1], bottom_right[2])
    top_right = (bottom_right[1], top_left[2])
    
    top_left = convert.(T, round.(top_left .* image_ratio))
    bottom_right = convert.(T, round.(bottom_right .* image_ratio))
    bottom_left = convert.(T, round.(bottom_left .* image_ratio))
    top_right = convert.(T, round.(top_right .* image_ratio))
    
    return top_left, bottom_left, bottom_right, top_right
end

#=
    generate_validation_image(image_path::String, classifiers::String) -> AbstractArray
    
Generates a bounding box around the face of a random image.

# Arguments

- `image_path::AbstractString`: The path to images
- `classifiers::Array{HaarLikeObject, 1}`: List of classifiers/haar like features

# Returns

- `validation_image::AbstractArray`: The new image with a bounding box
=#
function generate_validation_image(image_path::String, classifiers::Vector{HaarLikeObject})
    
    # === TODO: THIS FUNCTION IS A WORK IN PROGRESS ===
    
    img = load_image(image_path)
    img_size = size(img)
    
    top_lefts = [c.top_left for c in classifiers]
    bottom_rights = [c.bottom_right for c in classifiers]
    x_coords = vcat([x[1] for x in top_lefts], [x[1] for x in bottom_rights])
    y_coords = vcat([y[2] for y in top_lefts], [y[2] for y in bottom_rights])
    min_x, max_x = extrema(x_coords)
    min_y, max_y = extrema(y_coords)
    top_left = min_x, min_y
    bottom_right = max_x, max_y
    
    box_dimensions = scale_box(top_left, bottom_right, (19, 19), img_size)
    
    return draw(load(image_path), Polygon([Point(box_dimensions[1]), Point(box_dimensions[2]), Point(box_dimensions[3]), Point(box_dimensions[4])]))
end
