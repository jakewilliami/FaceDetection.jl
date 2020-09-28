#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
#=
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
=#


println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
const FD = FaceDetection
using Printf: @printf
using Images: imresize
using Serialization: deserialize

println("...done")

function main(;
    smart_choose_feats::Bool=false, alt::Bool=false
)

	# we assume that `smart_choose_feats = true`
    main_path = dirname(dirname(@__FILE__))
    data_path = joinpath(main_path, "data")
    main_image_path = joinpath(main_path, "data", "main")
    alt_image_path = joinpath(main_path, "data", "alt")

    if alt
        # pos_testing_path = joinpath(alt_image_path, "testing", "pos")
        # neg_testing_path = joinpath(homedir(), "Desktop", "Assorted Personal Documents", "Wallpapers copy")
        pos_testing_path = joinpath(main_image_path, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$main_image_path/testset/faces/"
        neg_testing_path = joinpath(main_image_path, "testset", "non-faces")
    else
        pos_testing_path = joinpath(main_image_path, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$main_image_path/testset/faces/"
        neg_testing_path = joinpath(main_image_path, "testset", "non-faces")
    end

    # pos_testing_path = joinpath(data_path, "lizzie-testset", "faces")
    # neg_testing_path = joinpath(data_path, "lizzie-testset", "nonfaces")
	
	if ! isfile(joinpath(dirname(@__FILE__), "data", "haar-like_features"))
		error(throw("You do not have a data file.  Ensure you run \"write.jl\" to obtain your Haar-like features before running this script/"))
	end

	# read classifiers from file
	classifiers = deserialize(joinpath(dirname(@__FILE__), "data", "haar-like_features"))

    FD.notify_user("Loading test faces...")

    faces_testing = FD.load_images(pos_testing_path)[1]
    # faces_ii_testing = map(FD.to_integral_image, faces_testing)
    faces_ii_testing = map(FD.to_integral_image, faces_testing)
    println("...done. ", length(faces_testing), " faces loaded.")

    FD.notify_user("Loading test non-faces..")

    non_faces_testing = FD.load_images(neg_testing_path)[1]
    non_faces_ii_testing = map(FD.to_integral_image, non_faces_testing)
    println("...done. ", length(non_faces_testing), " non-faces loaded.\n")

    FD.notify_user("Testing selected classifiers...")
    correct_faces = 0
    correct_non_faces = 0

    # correct_faces = sum([FD._get_feature_vote(face, classifiers) for face in faces_ii_testing])
    # correct_non_faces = length(non_faces_testing) - sum([FD._get_feature_vote(nonFace, classifiers) for nonFace in non_faces_ii_testing])
    correct_faces = sum(FD.ensemble_vote_all(faces_ii_testing, classifiers))
    correct_non_faces = length(non_faces_testing) - sum(FD.ensemble_vote_all(non_faces_ii_testing, classifiers))
    correct_faces_percent = (float(correct_faces) / length(faces_testing)) * 100
    correct_non_faces_percent = (float(correct_non_faces) / length(non_faces_testing)) * 100

    faces_frac = string(correct_faces, "/", length(faces_testing))
    faces_percent = string("(", correct_faces_percent, "% of faces were recognised as faces)")
    non_faces_frac = string(correct_non_faces, "/", length(non_faces_testing))
    non_faces_percent = string("(", correct_non_faces_percent, "% of non-faces were identified as non-faces)")

    println("...done.\n")
    FD.notify_user("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", faces_frac, faces_percent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", non_faces_frac, non_faces_percent)
end

@time main(smart_choose_feats=true, alt=false)
