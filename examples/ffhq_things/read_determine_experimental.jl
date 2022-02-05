# Adapted from https://github.com/Simon-Hohberg/Viola-Jones/

# Faces dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset/) 70_001 images of faces
# Non-faces dataset: [THINGS](https://osf.io/3fu6z/); 26_107 object images

@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

using FaceDetection
using Printf: @printf
using Images: imresize
using Serialization: deserialize

@info("...done")

function main(
    classifiers_file::String;
	smart_choose_feats::Bool=false,
	scale::Bool=true,
	scale_to::Tuple=(128, 128)
)
    
    @info("Calculating test face scores and constructing dataset...")
    sleep(0.5)
    
    data_path = joinpath(dirname(dirname(@__DIR__)), "data")
	
	pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
	neg_training_path = joinpath(data_path, "things", "object_images")
    testing_path = joinpath(data_path, "lizzie-testset", "2021")
	data_path = joinpath(dirname(dirname(@__DIR__)), "data")
	
    classifiers = deserialize(classifiers_file)
    sort!(classifiers, by = c -> c.weight, rev = true)

    @info("Testing selected classifiers...")
	sleep(3) # sleep here because sometimes the threads from `learn` are still catching up and then `ensemble_vote_all` errors
	
    face_testing_images       = readdir(joinpath(testing_path, "Faces"), join=true, sort=false)
    # face_testing_images       = readdir(joinpath(data_path, "lizzie-testset", "2022-unshined", "Selected faces"), join=true, sort=false)
    # face_testing_images       = readdir(joinpath(data_path, "lizzie-testset", "2022-unshined", "All "), join=true, sort=false)
    pareidolia_testing_images = readdir(joinpath(testing_path, "Pareidolia"), join=true, sort=false)
    flower_testing_images     = readdir(joinpath(testing_path, "Flowers"), join=true, sort=false)
    object_testing_images     = readdir(joinpath(testing_path, "Objects"), join=true, sort=false)
    
	num_faces      = length(face_testing_images)
    num_pareidolia = length(pareidolia_testing_images)
    num_flowers    = length(flower_testing_images)
    num_objects    = length(object_testing_images)
    
    # aliases
    pos_testing_images = face_testing_images
    neg_testing_images = object_testing_images
    num_faces          = num_faces
    num_non_faces      = num_objects
    
    # determining how many were correctly classified
	correct_faces             = sum(ensemble_vote_all(pos_testing_images, classifiers, scale=scale, scale_to=scale_to))
	correct_non_faces         = num_non_faces - sum(ensemble_vote_all(neg_testing_images, classifiers, scale=scale, scale_to=scale_to))
	correct_faces_percent     = (correct_faces / num_faces) * 100
	correct_non_faces_percent = (correct_non_faces / num_non_faces) * 100

    faces_frac        = string(correct_faces, "/", num_faces)
    faces_percent     = string("(", correct_faces_percent, "% of faces were recognised as faces)")
    non_faces_frac    = string(correct_non_faces, "/", num_non_faces)
    non_faces_percent = string("(", correct_non_faces_percent, "% of non-faces were identified as non-faces)")

    @info("...done.\n")
    @info("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", faces_frac, faces_percent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", non_faces_frac, non_faces_percent)
end

# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_2000_pos_2000_neg_(128,128)_(100,100,30,30)")
data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_5000_pos_5000_neg_(128,128)_(100,100,30,30)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_1500_pos_1500_neg_(128,128)_(128,128,1,1)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_200_pos_200_neg_(128,128)_(70,70,50,50)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_200_pos_200_neg_(24,24)_(-1,-1,1,1)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_1000_pos_1000_neg_(24,24)_(-1,-1,1,1)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_2000_pos_2000_neg_(24,24)_(-1,-1,1,1)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_4000_pos_4000_neg_(24,24)_(-1,-1,1,1)")
@time main(data_file; smart_choose_feats = false, scale = true, scale_to = (128, 128))
