#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
#=
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
=#


println("Loading required libraries (it will take a moment to precompile if it is your first time doing this)...")


include("IntegralImage.jl")
include("AdaBoost.jl") # imports HaarLikeFeature.jl implicitly
include("Utils.jl")


mainPath = "/Users/jakeireland/FaceDetection.jl/"
mainImagePath = "$mainPath/data/main/"
altImagePath = "$mainPath/data/alt/"


function main(alt::Bool=false, imageReconstruction::Bool=false)
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

      numClassifiers = 2
      # For performance reasons restricting feature size
      minFeatureHeight = 8
      maxFeatureHeight = 10
      minFeatureWidth = 8
      maxFeatureWidth = 10


      println("Loading faces...")
      
      facesTraining = loadImages(posTrainingPath)
      facesIITraining = map(toIntegralImage, facesTraining) # list(map(...))
      println("...done. ", length(facesTraining), " faces loaded.\n\nLoading non-faces...")
      nonFacesTraining = loadImages(negTrainingPath)
      nonFacesIITraining = map(toIntegralImage, nonFacesTraining) # list(map(...))
      println("...done. ", length(nonFacesTraining), " non-faces loaded.\n")

      # classifiers are haar like features
      classifiers = learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

      println("\nLoading test faces...")
      facesTesting = loadImages(posTestingPath)
      facesIITesting = map(toIntegralImage, facesTesting)
      println("...done. ", length(facesTesting), " faces loaded.\n\nLoading test non-faces..")
      nonFacesTesting = loadImages(negTestingPath)
      nonFacesIITesting = map(toIntegralImage, nonFacesTesting)
      println("...done. ", length(nonFacesTesting), " non-faces loaded.\n")

      println("Testing selected classifiers...")
      correctFaces = 0
      correctNonFaces = 0
      correctFaces = deepsum(ensembleVoteAll(facesIITesting, classifiers))
      correctNonFaces = length(nonFacesTesting) - deepsum(ensembleVoteAll(nonFacesIITesting, classifiers))
      correctFacesPercent = (deepfloat(correctFaces) / length(facesTesting)) * 100
      correctNonFacesPercent = (deepfloat(correctNonFaces) / length(nonFacesTesting)) * 100

      println("...done.\n\nResult:\n      Faces: ", correctFaces, "/", length(facesTesting), "  (", correctFacesPercent, "%)\n  non-Faces: ", correctNonFaces, "/", length(nonFacesTesting), "  (", correctNonFacesPercent, "%)")

      # Just for fun: putting all Haar-like features over each other generates a face-like image
      reconstructedImage = reconstruct(classifiers, size(facesTesting[1]))
      
      if imageReconstruction
            save(joinpath(homedir(), "Desktop", "reconstruction.png"), Gray.(map(clamp01nan, reconstructedImage)))
      end
end



main()
