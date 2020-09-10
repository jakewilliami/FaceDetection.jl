#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(realpath $(dirname $0))))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest

println("...done")


function main(; smartChooseFeats::Bool=false, alt::Bool=false)
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
    
    notifyUser("Calculating test face scores and constructing dataset...")
    
    # get scores
    # facesScores = Matrix{Float64}(undef, length(facesIITesting), 1)
    # nonFacesScores = Matrix{Float64}(undef, length(nonFacesIITesting), 1)
    facesScores = zeros(length(facesIITesting))
    nonFacesScores = zeros(length(nonFacesIITesting))
    
    facesScores[:] .= [sum([FaceDetection.getFaceness(c,face) for c in classifiers]) for face in facesIITesting]
    nonFacesScores[:] .= [sum([FaceDetection.getFaceness(c,nonFace) for c in classifiers]) for nonFace in nonFacesIITesting]
    
    # filling in the dataset with missing to easily write to csv
    dfFaces = facesScores
    dfNonFaces = nonFacesScores
    if length(facesScores) < length(nonFacesScores)
        toAdd = length(nonFacesIITesting) - length(facesIITesting)
        dfFaces = vcat(dfFaces, Matrix{Union{Float64, Missing}}(undef, toAdd, 1))
        faceNames = vcat(faceNames, Matrix{Union{Float64, Missing}}(undef, toAdd, 1))
    elseif length(facesScores) > length(nonFacesScores)
        length(facesIITesting) - length(nonFacesIITesting)
        dfNonFaces = vcat(dfNonFaces, Matrix{Union{Float64, Missing}}(undef, toAdd, 1))
        nonFaceNames = vcat(nonFaceNames, Matrix{Union{Float64, Missing}}(undef, toAdd, 1))
    end
    
    # write score data
    write(joinpath(dirname(dirname(@__FILE__)), "data", "faceness-scores.csv"), DataFrame(hcat(faceNames, dfFaces, nonFaceNames, dfNonFaces)), writeheader=false)
    
    println("...done.\n")
    
    notifyUser("Computing differences in scores between faces and non-faces...")
    
    welch_t = UnequalVarianceTTest(facesScores, nonFacesScores)
    
    println("...done.  $welch_t\n")
    
    notifyUser("Constructing box plot with said dataset...")
    
    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(facesScores, xaxis=false, label = false),
                    StatsPlots.boxplot(nonFacesScores, xaxis=false, label = false),
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


@time main(smartChooseFeats=true, alt=false)
