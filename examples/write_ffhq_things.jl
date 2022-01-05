# Adapted from https://github.com/Simon-Hohberg/Viola-Jones/

# Faces dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset/) 70_001 images of faces
# Non-faces dataset: [THINGS](https://osf.io/3fu6z/); 26_107 object images

@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

include(joinpath(dirname(@__DIR__), "src", "FaceDetection.jl"))

using .FaceDetection
using Serialization: serialize

@info("...done")

function takerand!(list::Vector{T}) where {T}
    j = rand(1:length(list))
    rand_elem = list[j]
    deleteat!(list, j)
    return rand_elem
end

rand_subset!(list::Vector{T}, n::Int) where {T} = 
    String[takerand!(list) for _ in 1:n]

"Return a random subset of the contents of directory `path` of size `n`."
function rand_subset_ls(path::String, n::Int)
	dir_contents = readdir(path, join=true, sort=false)
	filter!(f -> !occursin(r".*\.DS_Store", f), dir_contents)
	@assert(length(dir_contents) >= n, "Not enough files in given directory to select `n` random.")
	
    return rand_subset!(dir_contents, n)
end

function main(
	num_pos::Int,
	num_neg::Int;
	scale::Bool=true,
	scale_to::Tuple=(128, 128)
)
    data_path = joinpath(dirname(@__DIR__), "data")
	
    pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
    neg_training_path = joinpath(data_path, "things", "object_images")

    all_pos_images = rand_subset_ls(pos_training_path, 2num_pos)
    all_neg_images = rand_subset_ls(neg_training_path, 2num_neg)

    pos_training_images = all_pos_images[1:num_pos]
    neg_training_images = all_neg_images[1:num_neg]
	
    num_classifiers = 10
	
    # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (67, 67, 65, 65)
    # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (70, 70, 50, 50)
    max_feature_width, max_feature_height, min_feature_height, min_feature_width = (100, 100, 30, 30)
    min_size_img = (128, 128)

    # classifiers are haar like features
    classifiers = learn(pos_training_images, neg_training_images, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = scale, scale_to = scale_to)

    data_file = joinpath(@__DIR__, "data", "2021", "classifiers_$(num_classifiers)_from_$(num_pos)_pos_$(num_neg)_neg_($(join(min_size_img, ',')))_($max_feature_width,$max_feature_height,$min_feature_height,$min_feature_width)")
    serialize(data_file, classifiers)
end

@time main(2000, 2000; scale = true, scale_to = (128, 128))
