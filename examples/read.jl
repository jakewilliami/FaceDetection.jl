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

	include("constants.jl")

	if ! alt
		include("main_data.jl")
	else
		include("alt_data.jl")
	end
	
	if ! isfile(data_file)
		error(throw("You do not have a data file.  Ensure you run \"write.jl\" to obtain your Haar-like features before running this script/"))
	end

	# read classifiers from file
	classifiers = deserialize(data_file)

	FD.notify_user("Testing selected classifiers...")
	num_faces = length(filtered_ls(pos_testing_path))
	num_non_faces = length(filtered_ls(neg_testing_path))
	
	correct_faces = sum(FD.ensemble_vote_all(pos_testing_path, classifiers, scale=scale, scale_to=scale_to))
	correct_non_faces = num_non_faces - sum(FD.ensemble_vote_all(neg_testing_path, classifiers, scale=scale, scale_to=scale_to))
	correct_faces_percent = (correct_faces / num_faces) * 100
	correct_non_faces_percent = (correct_non_faces / num_non_faces) * 100

    faces_frac = string(correct_faces, "/", num_faces)
    faces_percent = string("(", correct_faces_percent, "% of faces were recognised as faces)")
    non_faces_frac = string(correct_non_faces, "/", num_non_faces)
    non_faces_percent = string("(", correct_non_faces_percent, "% of non-faces were identified as non-faces)")

    println("...done.\n")
    FD.notify_user("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", faces_frac, faces_percent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", non_faces_frac, non_faces_percent)
end

@time main(smart_choose_feats=true, alt=false)
