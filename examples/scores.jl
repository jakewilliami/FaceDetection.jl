#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
const FD = FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest
using Serialization: deserialize

println("...done")

function main(;
	smart_choose_feats::Bool=false,
	scale::Bool=false,
	scale_to::Tuple=(200, 200))
	
	include("constants.jl")
	include("main_data.jl")
    pos_testing_path = "/Users/jakeireland/projects/FaceDetection.jl/data/lizzie-testset/faces/"
	neg_testing_path = "/Users/jakeireland/projects/FaceDetection.jl/data/lizzie-testset/nonfaces/"
    
    # read classifiers from file
	classifiers = deserialize("/Users/jakeireland/Desktop/classifiers_100_577_577_fixed_1idx")
    num_classifiers = length(classifiers)
	
    notify_user("Calculating test face scores and constructing dataset...")
    
    faces_scores = Vector{Real}(undef, length(filtered_ls(pos_testing_path)))
    non_faces_scores = Vector{Real}(undef, length(filtered_ls(neg_testing_path)))
    
    faces_scores[:] .= [sum([FD.get_faceness(c, load_image(face, scale=scale, scale_to=scale_to)) for c in classifiers]) / num_classifiers for face in filtered_ls(pos_testing_path)]
	non_faces_scores[:] .= [sum([FD.get_faceness(c, load_image(non_face, scale=scale, scale_to=scale_to)) for c in classifiers]) / num_classifiers for non_face in filtered_ls(neg_testing_path)]
	
	face_names = basename.(filtered_ls(pos_testing_path))
	non_face_names = basename.(filtered_ls(neg_testing_path))
    
    # filling in the dataset with missing to easily write to csv
    df_faces = faces_scores
    df_non_faces = non_faces_scores
    if length(faces_scores) < length(non_faces_scores)
        to_add = length(filtered_ls(neg_testing_path)) - length(filtered_ls(pos_testing_path))
        df_faces = vcat(df_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        face_names = vcat(face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    elseif length(faces_scores) > length(non_faces_scores)
        to_add = length(filtered_ls(pos_testing_path)) - length(filtered_ls(neg_testing_path))
        df_non_faces = vcat(df_non_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        non_face_names = vcat(non_face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    end
    
    # write score data
	data_file = joinpath(dirname(dirname(@__FILE__)), "data", "faceness-scores.csv")
    write(data_file, DataFrame(hcat(face_names, df_faces, non_face_names, df_non_faces)), writeheader=false)
    
    println("...done.  Dataset written to $(data_file).\n")
    
    notify_user("Computing differences in scores between faces and non-faces...")
    
    welch_t = UnequalVarianceTTest(faces_scores, non_faces_scores)
    
    println("...done.  $welch_t\n")
    
    notify_user("Constructing box plot with said dataset...")
    
    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(faces_scores, xaxis=false, label = false),
                    StatsPlots.boxplot(non_faces_scores, xaxis=false, label = false),
                    title = ["Scores of Faces" "Scores of Non-Faces"],
                    # label = ["faces" "non-faces"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    # fillcolor = [:blue, :orange],
                    link = :y,
                    # framestyle = [:origin :origin]
                )
    
    # save plot
    StatsPlots.savefig(plot, joinpath(dirname(dirname(@__FILE__)), "figs", "scores.pdf"))
    
    println("...done.  Plot created at ", joinpath(dirname(dirname(@__FILE__)), "figs", "scores.pdf"), "\n")
end

@time main(smart_choose_feats=true, scale=true, scale_to=(577, 577))
