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
using Serialization: serialize

println("...done")

function main(;
    smart_choose_feats::Bool=false,
	alt::Bool=false,
	scale::Bool=false,
	scale_to::Tuple=(200, 200)
)
	include("constants.jl")

	if ! alt
		include("main_data.jl")
	else
		include("alt_data.jl")
	end

	min_size_img = (19, 19) # default for our test dataset
	if smart_choose_feats
		# For performance reasons restricting feature size
		notify_user("Selecting best feature width and height...")
		
		max_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = determine_feature_size(pos_training_path, neg_training_path; scale = scale, scale_to = scale_to)
		
		println("...done.  Maximum feature width selected is $max_feature_width pixels; minimum feature width is $min_feature_width; maximum feature height is $max_feature_height pixels; minimum feature height is $min_feature_height.\n")
	else
		min_feature_height = 8
		max_feature_height = 10
		min_feature_width = 8
		max_feature_width = 10
	end

	# classifiers are haar like features
	classifiers = FD.learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = scale, scale_to = scale_to)

	# write classifiers to file
	data_file = joinpath(dirname(@__FILE__), "data", "haar-like_features_c$(num_classifiers)")
	serialize(data_file, classifiers)
end

@time main(smart_choose_feats=true, alt=false, scale=true, scale_to=(20, 20))
