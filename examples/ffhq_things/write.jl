# Adapted from https://github.com/Simon-Hohberg/Viola-Jones/

# Faces dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset/) 70_001 images of faces
# Non-faces dataset: [THINGS](https://osf.io/3fu6z/); 26_107 object images

@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

include(joinpath(dirname(dirname(@__DIR__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Serialization: serialize
using Random: shuffle

@info("...done")

function takerand!(list::Vector{T}) where {T}
    j = rand(1:length(list))
    rand_elem = list[j]
    deleteat!(list, j)
    return rand_elem
end

rand_subset!(list::Vector{T}, n::Int) where {T} = String[takerand!(list) for _ in 1:n]

"Return a random subset of the contents of directory `path` of size `n`."
function rand_subset_ls(path::String, n::Int)
    dir_contents = readdir(path, join = true, sort = false)
    filter!(f -> !occursin(r".*\.DS_Store", f), dir_contents)

    if n == -1
        return shuffle(dir_contents)
    end

    @assert(
        length(dir_contents) >= n,
        "Not enough files in given directory to select `n` random."
    )

    return rand_subset!(dir_contents, n)
end

function main(num_pos::Int, num_neg::Int; scale::Bool = true, scale_to::Tuple = (128, 128))
    data_path = joinpath(dirname(dirname(@__DIR__)), "data")

    pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
    # pos_training_path = joinpath(data_path, "main", "trainset", "faces")
    # pos_training_path = joinpath(data_path, "alt", "pos")

    neg_training_path = joinpath(data_path, "things", "object_images")
    # neg_training_path = joinpath(data_path, "all-non-faces")
    # neg_training_path = joinpath(data_path, "main", "trainset", "non-faces")
    # neg_training_path = joinpath(data_path, "alt", "neg")

    pos_training_images = rand_subset_ls(pos_training_path, num_pos)
    neg_training_images = rand_subset_ls(neg_training_path, num_neg)
    println([basename(p) for p in pos_training_images])

    num_classifiers = 10

    # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (67, 67, 65, 65)
    # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (70, 70, 50, 50)
    # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (100, 100, 30, 30)
    max_feature_width, max_feature_height, min_feature_height, min_feature_width =
        (-1, -1, 1, 1)

    sleep(3)

    # classifiers are haar like features
    classifiers = learn(
        pos_training_images,
        neg_training_images,
        num_classifiers,
        min_feature_height,
        max_feature_height,
        min_feature_width,
        max_feature_width;
        scale = scale,
        scale_to = scale_to,
    )

    data_file = joinpath(
        dirname(@__DIR__),
        "data",
        "classifiers_$(num_classifiers)_from_$(num_pos)_pos_$(num_neg)_neg_($(join(scale_to, ',')))_($max_feature_width,$max_feature_height,$min_feature_height,$min_feature_width)",
    )
    serialize(data_file, classifiers)
end

@time main(8000, 8000; scale = true, scale_to = (24, 24))
