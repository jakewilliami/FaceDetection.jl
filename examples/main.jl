#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(realpath $(dirname $0))))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
#=
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
=#


println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Printf: @printf
using Images: imresize

println("...done")


function main(; smartChooseFeats::Bool=false, alt::Bool=false)
    # we assume that `smartChooseFeats = true`
    mainPath = dirname(dirname(@__FILE__))
    dataPath = joinpath(mainPath, "data")
    mainImagePath = joinpath(mainPath, "data", "main")
    altImagePath = joinpath(mainPath, "data", "alt")

    if alt
        posTrainingPath = joinpath(altImagePath, "pos")
        negTrainingPath = joinpath(altImagePath, "neg")
        # posTestingPath = joinpath(altImagePath, "testing", "pos")
        # negTestingPath = joinpath(homedir(), "Desktop", "Assorted Personal Documents", "Wallpapers copy")
        posTestingPath = joinpath(mainImagePath, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$mainImagePath/testset/faces/"
        negTestingPath = joinpath(mainImagePath, "testset", "non-faces")
    else
        posTrainingPath = joinpath(mainImagePath, "trainset", "faces")
        negTrainingPath = joinpath(mainImagePath, "trainset", "non-faces")
        posTestingPath = joinpath(mainImagePath, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$mainImagePath/testset/faces/"
        negTestingPath = joinpath(mainImagePath, "testset", "non-faces")
    end
    
    # posTrainingPath = joinpath(dataPath, "lfw-all")
    # negTrainingPath = joinpath(dataPath, "all-non-faces")
    # posTestingPath = joinpath(dataPath, "lizzie-testset", "faces")
    # negTestingPath = joinpath(dataPath, "lizzie-testset", "nonfaces")

    numClassifiers = 10

    minSizeImg = (19, 19) # default for our test dataset
    if smartChooseFeats
        # For performance reasons restricting feature size
        notifyUser("Selecting best feature width and height...")
        
        maxFeatureWidth, maxFeatureHeight, minFeatureHeight, minFeatureWidth, minSizeImg = determineFeatureSize(posTrainingPath, negTrainingPath)
        
        println("...done.  Maximum feature width selected is $maxFeatureWidth pixels; minimum feature width is $minFeatureWidth; maximum feature height is $maxFeatureHeight pixels; minimum feature height is $minFeatureHeight.\n")
    else
        minFeatureHeight = 8
        maxFeatureHeight = 10
        minFeatureWidth = 8
        maxFeatureWidth = 10
    end


    FaceDetection.notifyUser("Loading faces...")

    facesTraining, trainingFaceNames = FaceDetection.loadImages(posTrainingPath)
    facesIITraining = map(FaceDetection.toIntegralImage, facesTraining) # list(map(...))
    println("...done. ", length(facesTraining), " faces loaded.")

    FaceDetection.notifyUser("Loading non-faces...")

    nonFacesTraining, trainingNonFaceNames = FaceDetection.loadImages(negTrainingPath)
    nonFacesIITraining = map(FaceDetection.toIntegralImage, nonFacesTraining) # list(map(...))
    println("...done. ", length(nonFacesTraining), " non-faces loaded.\n")

    # classifiers are haar like features
    classifiers = FaceDetection.learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

    FaceDetection.notifyUser("Loading test faces...")

    facesTesting, faceNames = FaceDetection.loadImages(posTestingPath)
    # facesIITesting = map(FaceDetection.toIntegralImage, facesTesting)
    facesIITesting = map(FaceDetection.toIntegralImage, facesTesting)
    println("...done. ", length(facesTesting), " faces loaded.")

    FaceDetection.notifyUser("Loading test non-faces..")

    nonFacesTesting, nonFaceNames = FaceDetection.loadImages(negTestingPath)
    nonFacesIITesting = map(FaceDetection.toIntegralImage, nonFacesTesting)
    println("...done. ", length(nonFacesTesting), " non-faces loaded.\n")

    FaceDetection.notifyUser("Testing selected classifiers...")
    correctFaces = 0
    correctNonFaces = 0

    # correctFaces = sum([FaceDetection._get_feature_vote(face, classifiers) for face in facesIITesting])
    # correctNonFaces = length(non_faces_testing) - sum([FaceDetection._get_feature_vote(nonFace, classifiers) for nonFace in nonFacesIITesting])
    correctFaces = sum(FaceDetection.ensembleVoteAll(facesIITesting, classifiers))
    correctNonFaces = length(nonFacesTesting) - sum(FaceDetection.ensembleVoteAll(nonFacesIITesting, classifiers))
    correctFacesPercent = (float(correctFaces) / length(facesTesting)) * 100
    correctNonFacesPercent = (float(correctNonFaces) / length(nonFacesTesting)) * 100

    facesFrac = string(correctFaces, "/", length(facesTesting))
    facesPercent = string("(", correctFacesPercent, "% of faces were recognised as faces)")
    nonFacesFrac = string(correctNonFaces, "/", length(nonFacesTesting))
    nonFacesPercent = string("(", correctNonFacesPercent, "% of non-faces were identified as non-faces)")

    println("...done.\n")
    FaceDetection.notifyUser("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", facesFrac, facesPercent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", nonFacesFrac, nonFacesPercent)
end



@time main(smartChooseFeats=true, alt=false)
