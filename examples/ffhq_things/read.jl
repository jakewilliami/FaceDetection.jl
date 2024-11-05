# Adapted from https://github.com/Simon-Hohberg/Viola-Jones/

# Faces dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset/) 70_001 images of faces
# Non-faces dataset: [THINGS](https://osf.io/3fu6z/); 26_107 object images

@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

using FaceDetection
using Printf: @printf
using Images: imresize
using Serialization: deserialize

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
    dir_contents = readdir(path; join = true, sort = false)
    filter!(f -> !occursin(r".*\.DS_Store", f), dir_contents)
    @assert(
        length(dir_contents) >= n,
        "Not enough files in given directory to select `n` random."
    )

    return rand_subset!(dir_contents, n)
end

function main(
    classifiers_file::String,
    num_pos::Int,
    num_neg::Int;
    smart_choose_feats::Bool = false,
    scale::Bool = true,
    scale_to::Tuple = (128, 128),
)
    data_path = joinpath(dirname(dirname(@__DIR__)), "data")

    pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
    neg_training_path = joinpath(data_path, "things", "object_images")

    all_pos_images = rand_subset_ls(pos_training_path, 2num_pos)
    all_neg_images = rand_subset_ls(neg_training_path, 2num_neg)

    pos_training_images = all_pos_images[1:num_pos]
    neg_training_images = all_neg_images[1:num_neg]

    classifiers = deserialize(classifiers_file)

    @info("Testing selected classifiers...")
    sleep(3) # sleep here because sometimes the threads from `learn` are still catching up and then `ensemble_vote_all` errors

    pos_testing_images = all_pos_images[(num_pos + 1):(2num_pos)]
    neg_testing_images = all_neg_images[(num_neg + 1):(2num_neg)]
    num_faces = length(pos_testing_images)
    num_non_faces = length(neg_testing_images)

    correct_faces = sum(ensemble_vote_all(
        pos_testing_images, classifiers; scale = scale, scale_to = scale_to
    ))
    correct_non_faces =
        num_non_faces - sum(ensemble_vote_all(
            neg_testing_images, classifiers; scale = scale, scale_to = scale_to
        ),)
    correct_faces_percent = (correct_faces / num_faces) * 100
    correct_non_faces_percent = (correct_non_faces / num_non_faces) * 100

    faces_frac = string(correct_faces, "/", num_faces)
    faces_percent = string(
        "(", correct_faces_percent, "% of faces were recognised as faces)"
    )
    non_faces_frac = string(correct_non_faces, "/", num_non_faces)
    non_faces_percent = string(
        "(", correct_non_faces_percent, "% of non-faces were identified as non-faces)"
    )

    @info("...done.\n")
    @info("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", faces_frac, faces_percent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", non_faces_frac, non_faces_percent)
end

data_file = joinpath(
    dirname(@__DIR__),
    "data",
    "classifiers_10_from_2000_pos_2000_neg_(128,128)_(100,100,30,30)",
)
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_5000_pos_5000_neg_(128,128)_(100,100,30,30)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_1500_pos_1500_neg_(128,128)_(128,128,1,1)")
@time main(
    data_file, 2000, 2000; smart_choose_feats = false, scale = true, scale_to = (128, 128)
)
