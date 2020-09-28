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
using Images: Gray, clamp01nan, save, imresize, load

println("...done")

function main(;
    smart_choose_feats::Bool=false,
    alt::Bool=false,
    image_reconstruction::Bool=true,
    feat_validation::Bool=true
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

    faces_testing = FD.load_images(pos_testing_path)[1]
    # faces_ii_testing = map(FD.to_integral_image, faces_testing)
    faces_ii_testing = map(FD.to_integral_image, faces_testing)
    println("...done. ", length(faces_testing), " faces loaded.")

    FD.notify_user("Loading test non-faces..")

    non_faces_testing = FD.load_images(neg_testing_path)[1]
    non_faces_ii_testing = map(FD.to_integral_image, non_faces_testing)
    println("...done. ", length(non_faces_testing), " non-faces loaded.\n")

    if image_reconstruction
        # Just for fun: putting all Haar-like features over each other generates a face-like image
        FD.notify_user("Constructing an image of all Haar-like Features found...")
        
        reconstructed_image = FD.reconstruct(classifiers, size(faces_testing[1]))
        save(joinpath(dirname(dirname(@__FILE__)), "figs", "reconstruction.png"), Gray.(map(clamp01nan, reconstructed_image)))
        
        println("...done.  See ", joinpath(dirname(dirname(@__FILE__)), "figs", "reconstruction.png"), ".\n")
    end

    if feat_validation
        FD.notify_user("Constructing a validation image on a random image...")
        
        FD.generate_validation_image(FD.get_random_image(joinpath(dirname(dirname(@__FILE__)), "figs", "validation.png")), classifiers)
        
        println("...done.  See ", joinpath(dirname(dirname(@__FILE__)), "figs", "validation.png"), ".\n")
    end
end

@time main(smart_choose_feats=true, alt=false, image_reconstruction=true, feat_validation=true)
