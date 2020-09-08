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


function main(smartChooseFeats::Bool=false, alt::Bool=false)
    mainPath = dirname(dirname(@__FILE__))
    mainImagePath = joinpath(mainPath, "data", "main")
    altImagePath = joinpath(mainPath, "data", "alt")

    if alt
        posTrainingPath = joinpath(altImagePath, "pos")
        negTrainingPath = joinpath(altImagePath, "neg")
        posTestingPath = joinpath(altImagePath, "testing", "pos")
        negTestingPath = joinpath(homedir(), "Desktop", "Assorted Personal Documents", "Wallpapers copy")
    else
        posTrainingPath = joinpath(mainImagePath, "trainset", "faces")
        negTrainingPath = joinpath(mainImagePath, "trainset", "non-faces")
        posTestingPath = joinpath(mainImagePath, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$mainImagePath/testset/faces/"
        negTestingPath = joinpath(mainImagePath, "testset", "non-faces")
    end

    numClassifiers = 4

    if ! smartChooseFeats
        # For performance reasons restricting feature size
        minFeatureHeight = 8
        maxFeatureHeight = 10
        minFeatureWidth = 8
        maxFeatureWidth = 10
    end


    FaceDetection.notifyUser("Loading faces...")

    facesTraining = FaceDetection.loadImages(posTrainingPath)
    facesIITraining = map(FaceDetection.toIntegralImage, facesTraining) # list(map(...))
    println("...done. ", length(facesTraining), " faces loaded.")

    FaceDetection.notifyUser("Loading non-faces...")

    nonFacesTraining = FaceDetection.loadImages(negTrainingPath)
    nonFacesIITraining = map(FaceDetection.toIntegralImage, nonFacesTraining) # list(map(...))
    println("...done. ", length(nonFacesTraining), " non-faces loaded.\n")

    # classifiers are haar like features
    classifiers = FaceDetection.learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

    FaceDetection.notifyUser("Loading test faces...")

    facesTesting = FaceDetection.loadImages(posTestingPath)
    # facesIITesting = map(FaceDetection.toIntegralImage, facesTesting)
    facesIITesting = map(i -> imresize(i, (19,19)), map(FaceDetection.toIntegralImage, facesTesting))
    println("...done. ", length(facesTesting), " faces loaded.")

    FaceDetection.notifyUser("Loading test non-faces..")

    nonFacesTesting = FaceDetection.loadImages(negTestingPath)
    nonFacesIITesting = map(i -> imresize(i, (19,19)), map(FaceDetection.toIntegralImage, nonFacesTesting))
    println("...done. ", length(nonFacesTesting), " non-faces loaded.\n")
    
    notifyUser("Calculating test face scores and constructing dataset...")
    
    # get scores
    # facesScores = Matrix{Float64}(undef, length(facesIITesting), 1)
    # nonFacesScores = Matrix{Float64}(undef, length(nonFacesIITesting), 1)
    facesScores = zeros(length(facesIITesting))
    nonFacesScores = zeros(length(nonFacesIITesting))
    
    facesScores[1:length(facesScores)] .= [sum([FaceDetection.getFaceness(c,face) for c in classifiers]) for face in facesIITesting]
    nonFacesScores[1:length(nonFacesScores)] .= [sum([FaceDetection.getFaceness(c,nonFace) for c in classifiers]) for nonFace in nonFacesIITesting]
    
    # filling in the dataset with missing to easily write to csv
    dfFaces = facesScores
    dfNonFaces = nonFacesScores
    if length(facesScores) < length(nonFacesScores)
        dfFaces = vcat(dfFaces, Matrix{Union{Float64, Missing}}(undef, length(nonFacesIITesting) - length(facesIITesting), 1))
    elseif length(facesScores) > length(nonFacesScores)
        dfNonFaces = vcat(dfNonFaces, Matrix{Union{Float64, Missing}}(undef, length(facesIITesting) - length(nonFacesIITesting), 1))
    end
    
    # write score data
    write(joinpath(homedir(), "Desktop", "facelikeness-data.csv"), DataFrame(hcat(dfFaces, dfNonFaces)), writeheader=false)
    
    println("...done.\n")
    
    notifyUser("Computing differences in scores between faces and non-faces...")
    
    welch_t = UnequalVarianceTTest(facesScores, nonFacesScores)
    
    println("...done.  $welch_t\n")
    
    notifyUser("Constructing box plot with said dataset...")
    
    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(facesScores, xaxis=false)
                    StatsPlots.boxplot(nonFacesScores, xaxis=false),
                    title = ["Scores of Faces" "Scores of Non-Faces"],
                    label = ["faces" "non-faces"],
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



@time main(false, false)
