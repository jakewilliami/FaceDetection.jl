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
    smart_choose_feats::Bool=false, alt::Bool=false
)
    include("constants.jl")

    if ! alt
        include("main_data.jl")
    else
        include("alt_data.jl")
    end
    
    # read classifiers from file
	classifiers = deserialize(data_file)

    FD.notify_user("Loading test faces...")

    faces_testing, face_names = FD.load_images(pos_testing_path)
    # faces_ii_testing = map(FD.to_integral_image, faces_testing)
    faces_ii_testing = map(FD.to_integral_image, faces_testing)
    println("...done. ", length(faces_testing), " faces loaded.")

    FD.notify_user("Loading test non-faces..")

    non_faces_testing, non_face_names = FD.load_images(neg_testing_path)
    non_faces_ii_testing = map(FD.to_integral_image, non_faces_testing)
    println("...done. ", length(non_faces_testing), " non-faces loaded.\n")
    
    notify_user("Calculating test face scores and constructing dataset...")
    
    # get scores
    # faces_scores = Matrix{Float64}(undef, length(faces_ii_testing), 1)
    # non_faces_scores = Matrix{Float64}(undef, length(non_faces_ii_testing), 1)
    faces_scores = zeros(length(faces_ii_testing))
    non_faces_scores = zeros(length(non_faces_ii_testing))
    
    faces_scores[:] .= [sum([FD.get_faceness(c,face) for c in classifiers]) for face in faces_ii_testing]
    non_faces_scores[:] .= [sum([FD.get_faceness(c,nonFace) for c in classifiers]) for nonFace in non_faces_ii_testing]
    
    # filling in the dataset with missing to easily write to csv
    df_faces = faces_scores
    df_non_faces = non_faces_scores
    if length(faces_scores) < length(non_faces_scores)
        to_add = length(non_faces_ii_testing) - length(faces_ii_testing)
        df_faces = vcat(df_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        face_names = vcat(face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    elseif length(faces_scores) > length(non_faces_scores)
        length(faces_ii_testing) - length(non_faces_ii_testing)
        df_non_faces = vcat(df_non_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        non_face_names = vcat(non_face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    end
    
    # write score data
    write(joinpath(dirname(dirname(@__FILE__)), "data", "faceness-scores.csv"), DataFrame(hcat(face_names, df_faces, non_face_names, df_non_faces)), writeheader=false)
    
    println("...done.\n")
    
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

@time main(smart_choose_feats=true, alt=false)
