#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
    
"""
Adapted from https://github.com/Simon-Hohberg/Viola-Jones/
"""

println("Loading required libraries (it will take a moment to precompile if it is your first time doing this)...")

include("IntegralImage.jl")
include("AdaBoost.jl")
include("Utils.jl")

mainPath = "/Users/jakeireland/FaceDetection.jl/"
mainImagePath = "$mainPath/data/main/"
altImagePath = "$mainPath/data/alt/"


function main(alt::Bool=false, HaarLikeFeature::Bool=false)
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


      ### TRAINING PROCESS
      println("Loading faces...")
      
      facesTraining = loadImages(posTrainingPath)
      facesIITraining = map(toIntegralImage, facesTraining) # list(map(...))
      println("...done. ", length(facesTraining), " faces loaded.\n\nLoading non-faces...")
      nonFacesTraining = loadImages(negTrainingPath)
      nonFacesIITraining = map(toIntegralImage, nonFacesTraining) # list(map(...))
      println("...done. ", length(nonFacesTraining), " non-faces loaded.\n")
      
      # println("Writing arrays to files for testing...")
      # rm(joinpath(homedir(), "Desktop", "facesIITraining.txt"))
      # rm(joinpath(homedir(), "Desktop", "nonFacesIITraining.txt"))
      # open(joinpath(homedir(), "Desktop", "facesIITraining.txt"), "w") do io
      #      write(io, string(facesIITraining))
      # end
      # open(joinpath(homedir(), "Desktop", "nonFacesIITraining.txt"), "w") do io
      #      write(io, string(nonFacesIITraining))
      # end

      # classifiers are haar like features
      # println("Determining classifiers; this may take a while...")
      classifiers = learn(facesIITraining, nonFacesIITraining, numClassifiers, minFeatureHeight, maxFeatureHeight, minFeatureWidth, maxFeatureWidth)

      # [println(c) for c in classifiers]
      # println(typeof(classifiers))
      # for c in classifiers
      #       println(typeof(c))
      # end

      println("\nLoading test faces...")
      facesTesting = loadImages(posTestingPath)
      facesIITesting = map(toIntegralImage, facesTesting) # list(map(...))
      println("...done. ", length(facesTesting), " faces loaded.\n\nLoading test non-faces..")
      nonFacesTesting = loadImages(negTestingPath)
      nonFacesIITesting = map(toIntegralImage, nonFacesTesting) # list(map(...))
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
      # reconstructedImage = [reconstruct(c, size(facesTesting[1])) for c in classifiers]
      # reconstructedImage = map(c -> reconstruct(c, size(facesTesting[1])), classifiers)
      
      # recon.save("/Users/jakeireland/Desktop/reconstruction.png")
      
      # println(reconstructedImage)
      # save("/Users/jakeireland/Desktop/reconstruction.png", colorview(Gray, map(clamp01nan, reconstructedImage)))
      # save("/Users/jakeireland/Desktop/reconstruction.png", reconstructedImage)
      # readblob(reconstructedImage, "/Users/jakeireland/Desktop/reconstruction.png")
      
      if HaarLikeFeatures
      # save("/Users/jakeireland/Desktop/reconstruction.png", reconstructedImage)
            save(joinpath(homedir(), "/Desktop/reconstruction.png"), Gray.(map(clamp01nan, reconstructedImage)))
      # save("/Users/jakeireland/Desktop/reconstruction.png", Gray.(reconstructedImage ./ 255))
      # println(channelview(reconstructedImage))
      # colorview(Gray, reconstructedImage)
      end

end



main()
