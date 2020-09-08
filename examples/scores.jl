#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(realpath $(dirname $0))))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Images: imresize
using StatsPlots#, Plots # StatsPlots required for box plots
using CSV: write
using DataFrames: DataFrame


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

    FaceDetection.notifyUser("Testing selected classifiers...")
    correctFaces = 0
    correctNonFaces = 0
    correctFaces = sum(FaceDetection.ensembleVoteAll(facesIITesting, classifiers))
    correctNonFaces = length(nonFacesTesting) - sum(FaceDetection.ensembleVoteAll(nonFacesIITesting, classifiers))
    correctFacesPercent = (float(correctFaces) / length(facesTesting)) * 100
    correctNonFacesPercent = (float(correctNonFaces) / length(nonFacesTesting)) * 100

    println("...done.\n")
    
    notifyUser("Calculating test face scores and constructing dataset...")
    
    dfFaces = Matrix{Union{Float64, Missing}}(undef, length(facesIITesting), 1)
    dfNonFaces = Matrix{Union{Float64, Missing}}(undef, length(nonFacesIITesting), 1)
    dfFaces[1:length(facesIITesting)] .= [sum([FaceDetection.getFaceness(c,face) for c in classifiers]) for face in facesIITesting]
    dfNonFaces[1:length(nonFacesIITesting)] .= [sum([FaceDetection.getFaceness(c,nonFace) for c in classifiers]) for nonFace in nonFacesIITesting]
    
    
    # displaymatrix(dfFaces)
    # displaymatrix(dfNonFaces)
    
    println("...done.\n")
    
    notifyUser("Constructing box plot with said dataset...")
    
    theme(:solarized)
    plot = boxplot(["" ""],# titles?
                    dfFaces, dfNonFaces,
                    title = ["Scores of Faces" "Scores of Non-Faces"],
                    label = ["faces" "non-faces"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    # fillcolor = [:blue, :orange],
                    link = :y,
                    framestyle = [:origin :origin]
                )
                
    plot(
        boxplot(dfFaces,
                        title = "Scores of Faces",
                        label = "faces",
                        fontfamily = font("Times"),
                        # fillcolor = [:blue, :orange],
                        link = :y,
                        framestyle = [:origin :origin]
                    )
        boxplot(dfNonFaces,
                        title = "Scores of Non-Faces",
                        label = non-faces",
                        fontfamily = font("Times"),
                        framestyle = [:origin :origin]
                    )
    )
    
    if length(dfFaces) < length(dfNonFaces) # filling in the dataset
        dfFaces = vcat(dfFaces, Matrix{Union{Float64, Missing}}(undef, length(nonFacesIITesting) - length(facesIITesting), 1))
    elseif length(dfFaces) > length(dfNonFaces)
        dfNonFaces = vcat(dfNonFaces, Matrix{Union{Float64, Missing}}(undef, length(facesIITesting) - length(nonFacesIITesting), 1))
    end
    
    write(joinpath(homedir(), "Desktop", "facelikeness-data.csv"), DataFrame(hcat(dfFaces, dfNonFaces)), writeheader=false)
    
    savefig(plot, joinpath(dirname(dirname(@__FILE__)), "figs", "scores.pdf"))
    
    println("...done.  Plot created at ", joinpath(dirname(dirname(@__FILE__)), "figs", "scores.pdf"), "\n")

end



@time main(false, false)
