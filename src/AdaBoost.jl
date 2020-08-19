#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
using ProgressMeter
using Distributed # for parallel processing (namely, @everywhere)

include("HaarLikeFeature.jl")



# TODO: select optimal threshold for each feature
# TODO: attentional cascading


function learn(positiveIIs, negativeIIs, numClassifiers=-1, minFeatureWidth=1, maxFeatureWidth=-1, minFeatureHeight=1, maxFeatureHeight=-1)
    """
    Selects a set of classifiers. Iteratively takes the best classifiers based
    on a weighted error.
    :param positive_iis: List of positive integral image examples
    :type positive_iis: list[numpy.ndarray]
    :param negative_iis: List of negative integral image examples
    :type negative_iis: list[numpy.ndarray]
    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int
    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]
    """
    numPos = length(positiveIIs)
    numNeg = length(negativeIIs)
    global numImgs = numPos + numNeg
    imgHeight, imgWidth = size(positiveIIs[1])
    
    # Maximum feature width and height default to image width and height
    if isequal(maxFeatureHeight, -1)
        maxFeatureHeight = imgHeight
    end
    if isequal(maxFeatureWidth, -1)
        maxFeatureWidth = imgWidth
    end
    
    # Create initial weights and labels
    posWeights = ones(numPos) * 1. / (2 * numPos)
    negWeights = ones(numNeg) * 1. / (2 * numNeg)
    weights = hcat((posWeights, negWeights))
    labels = hcat((ones(numPos), ones(numNeg) * -1))
    
    global images = positiveIIs + negativeIIs
    # global images = vcat(positiveIIs, negativeIIs)

    # Create features for all sizes and locations
    features = _create_features(imgHeight, imgWidth, minFeatureWidth, maxFeatureWidth, minFeatureHeight, maxFeatureHeight)
    numFeatures = length(features)
    # feature_indexes = list(range(num_features))
    featureIndexes = Array(1:numFeatures)
    
    if isequal(numClassifiers, -1)
        numClassifiers = numFeatures
    end
    
    votes = zeros((numImgs, numFeatures))
    # bar = progressbar.ProgressBar()
    # @everywhere numImgs begin
    @everywhere begin
        n = numImgs
        processes = length(numImgs)
        p = Progress(n, 1)   # minimum update interval: 1 second
        for i in processes # bar(range(num_imgs)):
            # votes[i, :] = np.array(partial(_get_feature_vote, image=images[i]), features)
            votes[i, :] = Array(map(partial(_get_feature_vote)(image=images[i])), features)
            next!(p)
        end
    end # end everywhere (end parallel processing)
    
    # select classifiers
    classifiers = Array()

    println("Selecting classifiers...")
    
    n = numClassifiers
    p = Progress(n, 1)   # minimum update interval: 1 second
    for i in processes
        classificationErrors = zeros(length(featureIndexes))

        # normalize weights
        weights *= 1. / sum(weights)

        # select best classifier based on the weighted error
        for f in 1:length(feature_indexes)
            fIDX = featureIndexes[f]
            # classifier error is the sum of image weights where the classifier
            # is right
            # error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            error = sum(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, fIDX]) ? weights[imgIDX] : 0, 1:numImgs)
            classificationErrors[f] = error
        end

        # get best feature, i.e. with smallest error
        minErrorIDX = argmin(classificationErrors)
        bestError = classificationErrors[minErrorIDX]
        bestFeatureIDX = featureIndexes[minErrorIDX]

        # set feature weight
        bestFeature = features[bestFeatureIDX]
        featureWeight = 0.5 * log((1 - bestError) / bestError)
        bestFeature.weight = featureWeight

        classifiers = vcat(classifiers, bestFeature)

        # update image weights
        # weights = list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs)))
        weights = (imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX]*sqrt((1-bestError)/bestError) : weights[imgIDX]*sqrt(bestError/(1-bestError)), 1:numImgs)

        # remove feature (a feature can't be selected twice)
        # feature_indexes.remove(best_feature_idx)
        featureIndexes = filter!(e -> e ∉ bestFeatureIDX, featureIndexes) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
        
        next!(p)
    end
    
    return classifiers
    
end


function _get_feature_vote(feature, image)
    return vote(image)
end


function _create_features(imgHeight::Int64, imgWidth::Int64, minFeatureWidth::Int64, maxFeatureWidth::Int64, minFeatureHeight::Int64, maxFeatureHeight::Int64)
    println("Creating Haar-like features...")
    # features = Array()
    features = []
    
    for feature in FeatureTypes # from HaarLikeFeature.jl
        # FeatureTypes are just tuples
        println(typeof(feature), " " , feature)
        featureStartWidth = max(minFeatureWidth, feature[1])
        for featureWidth in range(featureStartWidth, stop=maxFeatureWidth, step=feature[1])
            featureStartHeight = max(minFeatureHeight, feature[2])
            for featureHeight in range(featureStartHeight, stop=maxFeatureHeight, step=feature[2])
                for x in 1:(imgWidth - featureWidth)
                    for y in 1:(imgHeight - featureHeight)
                        feature = vcat(feature, HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, 1))
                        feature = vcat(feature, HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, -1))
                    end # end for y
                end # end for x
            end # end for feature height
        end # end for feature width
    end # end for feature in feature types
    
    println("...finished processing; ", length(features), " features created.")
    
    return features
end


export learn
export _get_feature_vote
export _create_features



### TESTING
