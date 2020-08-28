#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
#=
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
=#

println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")

include("FaceDetection.jl")

using .FaceDetection
using Printf: @printf
using Images: Gray, clamp01nan, save


function main(alt::Bool=false, imageReconstruction::Bool=false)
      mainPath = "/Users/jakeireland/FaceDetection.jl/"
      mainImagePath = "$mainPath/data/main/"
      altImagePath = "$mainPath/data/alt/"
      
      if alt
            posTrainingPath = "$altImagePath/pos/"
            negTrainingPath = "$altImagePath/neg/"
            posTestingPath = "$altImagePath/testing/pos/"
            negTestingPath = "/Users/jakeireland/Desktop/Assorted Personal Documents/Wallpapers copy/"
      elseif ! alt
            posTrainingPath = "$mainImagePath/trainset/faces/"
            negTrainingPath = "$mainImagePath/trainset/non-faces/"
            posTestingPath = "$mainImagePath/testset/faces/"
            negTestingPath = "$mainImagePath/testset/non-faces/"
      end

      numClassifiers = 10
      # For performance reasons restricting feature size
      minFeatureHeight = 8
      maxFeatureHeight = 10
      minFeatureWidth = 8
      maxFeatureWidth = 10


      notifyUser("Loading faces...")
      
      facesTraining = loadImages(posTrainingPath)
      facesIITraining = map(toIntegralImage, facesTraining) # list(map(...))
      println("...done. ", length(facesTraining), " faces loaded.")
      
      notifyUser("Loading non-faces...")
      
      nonFacesTraining = loadImages(negTrainingPath)
      nonFacesIITraining = map(toIntegralImage, nonFacesTraining) # list(map(...))
      println("...done. ", length(nonFacesTraining), " non-faces loaded.\n")

      # classifiers are haar like features
      classifiers = learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

      notifyUser("Loading test faces...")
      
      facesTesting = loadImages(posTestingPath)
      facesIITesting = map(toIntegralImage, facesTesting)
      println("...done. ", length(facesTesting), " faces loaded.")
      
      notifyUser("Loading test non-faces..")
      
      nonFacesTesting = loadImages(negTestingPath)
      nonFacesIITesting = map(toIntegralImage, nonFacesTesting)
      println("...done. ", length(nonFacesTesting), " non-faces loaded.\n")

      notifyUser("Testing selected classifiers...")
      correctFaces = 0
      correctNonFaces = 0
      correctFaces = sum(ensembleVoteAll(facesIITesting, classifiers))
      correctNonFaces = length(nonFacesTesting) - sum(ensembleVoteAll(nonFacesIITesting, classifiers))
      correctFacesPercent = (float(correctFaces) / length(facesTesting)) * 100
      correctNonFacesPercent = (float(correctNonFaces) / length(nonFacesTesting)) * 100

      facesFrac = string(correctFaces, "/", length(facesTesting))
      facesPercent = string("(", correctFacesPercent, "% of faces were recognised as faces)")
      nonFacesFrac = string(correctNonFaces, "/", length(nonFacesTesting))
      nonFacesPercent = string("(", correctNonFacesPercent, "% of non-faces were identified as non-faces)")

      println("...done.\n")
      notifyUser("Result:\n")
      
      @printf("%10.9s %10.9s %15s\n", "Faces:", facesFrac, facesPercent)
      @printf("%10.9s %10.9s %15s\n\n", "Non-faces:", nonFacesFrac, nonFacesPercent)

      # Just for fun: putting all Haar-like features over each other generates a face-like image
      reconstructedImage = reconstruct(classifiers, size(facesTesting[1]))
      
      if imageReconstruction
            notifyUser("Constructing an image of all Haar-like Features found...")
            
            save(joinpath(homedir(), "Desktop", "reconstruction.png"), Gray.(map(clamp01nan, reconstructedImage)))
            
            println("...done.  See ", joinpath(homedir(), "Desktop", "reconstruction.png"), ".\n")
      end
end



@time main(false, true)
