#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
"""
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
"""


include("IntegralImage.jl")
include("AdaBoost.jl")
include("Utils.jl")


function main()
      posTrainingPath = "/Users/jakeireland/FaceDetection.jl/test/images/faces/"
      negTrainingPath = "/Users/jakeireland/FaceDetection.jl/test/images/nonfaces/"
      posTestingPath = "/Users/jakeireland/FaceDetection.jl/test/images/faces/test/"
      negTestingPath = "/Users/jakeireland/FaceDetection.jl/test/images/nonfaces/test/"


      num_classifiers = 2
      # For performance reasons restricting feature size
      min_feature_height = 8
      max_feature_height = 10
      min_feature_width = 8
      max_feature_width = 10


      println("Loading faces...")
      
      facesTraining = loadImages(posTrainingPath)
      facesIITraining = map(toIntegralImage, facesTraining) # list(map(...))
      println("...done. ", length(facesTraining), " faces loaded.\n\nLoading non-faces...")
      nonFacesTraining = loadImages(negTrainingPath)
      nonFacesIITraining = map(toIntegralImage, nonFacesTraining) # list(map(...))
      println("./.done. ", length(nonFacesTraining), " non-faces loaded.\n")

      # classifiers are haar like features
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
