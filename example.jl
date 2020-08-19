#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
"""
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
"""

println("Loading required libraries (it will take a moment to precompile is this is your first time doing this)...")

include("IntegralImage.jl")
include("AdaBoost.jl")
include("Utils.jl")


function main()
      posTrainingPath = "/Users/jakeireland/FaceDetection.jl/test/images/pos/"
      negTrainingPath = "/Users/jakeireland/FaceDetection.jl/test/images/neg/"
      posTestingPath = "/Users/jakeireland/FaceDetection.jl/test/images/testing/pos/"
      negTestingPath = "/Users/jakeireland/Desktop/Assorted Personal Documents/Wallpapers copy/"

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
      println("Determining classifiers; this will take a while...")
      classifiers = learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

      println("Loading test faces...")
      facesTesting = loadImages(posTestingPath)
      facesIITesting = map(toIntegralImage, facesTesting) # list(map(...))
      println("...done. ", length(facesTesting), " faces loaded.\n\nLoading test non-faces..")
      nonFacesTesting = load_images(negTestingPath)
      nonFacesIITesting = map(toIntegralImage, nonFacesTesting) # list(map(...))
      println("...done. ", length(nonFacesTesting), " non-faces loaded.\n")

      println("Testing selected classifiers...")
      correctFaces = 0
      correctNonFaces = 0
      correctFaces = sum(ensembleVoteAll(facesIITesting, classifiers))
      correctNonFaces = length(nonFacesTesting) - sum(ensembleVoteAll(nonFacesIITesting, classifiers))

      println("...done.\n\nResult:\n      Faces: ", correctFaces, "/", length(faces_testing)
            , "  (", ((float(correctFaces) / len(facesTesting)) * 100), "%)\n  non-Faces: "
            , (correct_non_faces), "/", len(non_faces_testing), "  ("
            , ((float(correct_non_faces) / len(non_faces_testing)) * 100), "%)")

      # Just for fun: putting all haar-like features over each other generates a face-like image
      recon = reconstruct(classifiers, size(facesTesting[1]))
      recon.save("/Users/jakeireland/Desktop/reconstruction.png")
end



main()
