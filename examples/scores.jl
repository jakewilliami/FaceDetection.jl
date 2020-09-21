#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(realpath $(dirname $0))))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
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

println("...done")

function main(;
    smart_choose_feats::Bool=false, alt::Bool=false
)
    main_path = dirname(dirname(@__FILE__))
    data_path = joinpath(main_path, "data")
    main_image_path = joinpath(main_path, "data", "main")
    alt_image_path = joinpath(main_path, "data", "alt")

    if alt
        pos_training_path = joinpath(alt_image_path, "pos")
        neg_training_path = joinpath(alt_image_path, "neg")
        # pos_testing_path = joinpath(alt_image_path, "testing", "pos")
        # neg_testing_path = joinpath(homedir(), "Desktop", "Assorted Personal Documents", "Wallpapers copy")
        pos_testing_path = joinpath(main_image_path, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$main_image_path/testset/faces/"
        neg_testing_path = joinpath(main_image_path, "testset", "non-faces")
    else
        pos_training_path = joinpath(main_image_path, "trainset", "faces")
        neg_training_path = joinpath(main_image_path, "trainset", "non-faces")
        pos_testing_path = joinpath(main_image_path, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$main_image_path/testset/faces/"
        neg_testing_path = joinpath(main_image_path, "testset", "non-faces")
    end
    
    # pos_training_path = joinpath(data_path, "lfw-all")
    # neg_training_path = joinpath(data_path, "all-non-faces")
    # pos_testing_path = joinpath(data_path, "lizzie-testset", "faces")
    # neg_testing_path = joinpath(data_path, "lizzie-testset", "nonfaces")

    num_classifiers = 10

    min_size_img = (19, 19) # default for our test dataset
    if smart_choose_feats
        # For performance reasons restricting feature size
        notify_user("Selecting best feature width and height...")
        
        max_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = determine_feature_size(pos_training_path, neg_training_path)
        
        println("...done.  Maximum feature width selected is $max_feature_width pixels; minimum feature width is $min_feature_width; maximum feature height is $max_feature_height pixels; minimum feature height is $min_feature_height.\n")
    else
        min_feature_height = 8
        max_feature_height = 10
        min_feature_width = 8
        max_feature_width = 10
    end


    FD.notify_user("Loading faces...")

    faces_training = FD.load_images(pos_training_path)[1]
    faces_ii_training = map(FD.to_integral_image, faces_training) # list(map(...))
    println("...done. ", length(faces_training), " faces loaded.")

    FD.notify_user("Loading non-faces...")

    non_faces_training = FD.load_images(neg_training_path)[1]
    non_faces_ii_training = map(FD.to_integral_image, non_faces_training) # list(map(...))
    println("...done. ", length(non_faces_training), " non-faces loaded.\n")

    # classifiers are haar like features
    classifiers = FD.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)

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
