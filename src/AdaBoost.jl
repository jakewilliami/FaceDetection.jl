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
    # Maximum feature width and height default to image width and height
    # max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    # max_feature_width = img_width if max_feature_width == -1 else max_feature_width
    if isequal(maxFeatureHeight, -1)
        maxFeatureHeight = imgHeight
    end
    if isequal(maxFeatureWidth, -1)
        maxFeatureWidth = imgWidth
    end
    
    # Create initial weights and labels
    posWeights = deepfloat(ones(numPos)) / (2 * numPos)
    negWeights = deepfloat(ones(numNeg)) / (2 * numNeg)
    #=
    Consider the original code,
        ```
        $ python3 -c 'import numpy; a=[1,2,3]; b=[4,5,6]; print(numpy.hstack((a,b)))'
        [1 2 3 4 5 6]
        ```
    This is *not* comparablt to Julia's `hcat`.  Here,
        ```
        numpy.hstack((a,b)) ≡ vcat(a,b)
        ```
    The series of `deep*` functions——though useful in general——were designed from awkward arrays of tuples of arrays, which came about from a translation error in this case.
    =#
    weights = vcat(posWeights, negWeights)
    # weights = vcat(posWeights, negWeights)
    # weights = hcat((posWeights, negWeights))
    # weights = vcat([posWeights, negWeights])
    # labels = vcat((ones(numPos), ones(numNeg) * -1))
    labels = vcat(ones(numPos), ones(numNeg) * -1)
    # labels = vcat([ones(numPos), ones(numNeg) * -1])
    
    global images = positiveIIs + negativeIIs
    # global images = vcat(positiveIIs, negativeIIs)

    # Create features for all sizes and locations
    global features = _create_features(imgHeight, imgWidth, minFeatureWidth, maxFeatureWidth, minFeatureHeight, maxFeatureHeight)
    numFeatures = length(features)
    # feature_indexes = list(range(num_features))
    featureIndices = Array(1:numFeatures)
    
    if isequal(numClassifiers, -1)
        numClassifiers = numFeatures
    end
    
    # println(typeof(numImgs));println(typeof(numFeatures))
    global votes = zeros((numImgs, numFeatures)) # necessarily different from `zero.((numImgs, numFeatures))`

    # bar = progressbar.ProgressBar()
    # @everywhere numImgs begin
    @everywhere begin
        n = numImgs
        processes = length(numImgs)
        p = Progress(n, 1)   # minimum update interval: 1 second
        for i in processes # bar(range(num_imgs)):
            # votes[i, :] = np.array(list(Pool(processes=None).map(partial(_get_feature_vote, image=images[i]), features)))
            # votes[i, :] = Array(map(partial(getVote, images[i]), features))
            votes[i, :] = Array(map(feature -> getVote(feature, images[i]), features))
            # votes[i, :] = [map(feature -> getVote(feature, images[i]), features)]
            next!(p)
        end
    end # end everywhere (end parallel processing)
    
    # select classifiers
    # classifiers = Array()
    classifiers = []

    println("Selecting classifiers...")
    
    n = numClassifiers
    p = Progress(n, 1)   # minimum update interval: 1 second
    for i in processes
        # println(typeof(length(featureIndices)))
        classificationErrors = zeros(length(featureIndices))

        # normalize weights
        # weights *= 1. / np.sum(weights)
        # weights *= float(1) / sum(weights)
        # weights = float(weights) / sum(weights) # ensure weights do not get rounded by element-wise making them floats
        # weights *= 1. / sum(weights)
        # weights = weights/sum(weights)
        # weights *= 1. / sum(weights)
        weights = deepdiv(deepfloat(weights), deepsum(weights))

        # select best classifier based on the weighted error
        for f in 1:length(featureIndices)
            fIDX = featureIndices[f]
            # classifier error is the sum of image weights where the classifier is right
            # error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            # error = sum(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, fIDX]) ? weights[imgIDX] : 0, 1:numImgs)
            # error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            error = deepsum(map(imgIDX -> labels[imgIDX] ≠ votes[imgIDX, fIDX] ? weights[imgIDX] : 0, 1:numImgs))
            # lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0
            # imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, fIDX]) ? weights[imgIDX] : 0
            
            classificationErrors[f] = error
        end

        # get the best feature; i.e. the feature with the smallest error
        minErrorIDX = argmin(classificationErrors) # returns the index of the minimum in the array
        bestError = classificationErrors[minErrorIDX]
        bestFeatureIDX = featureIndices[minErrorIDX]

        # set feature weight
        bestFeature = features[bestFeatureIDX]
        featureWeight = 0.5 * log((1 - bestError) / bestError)
        bestFeature.weight = featureWeight

        # classifiers = vcat(classifiers, bestFeature)
        classifiers = push!(classifiers, bestFeature)

        # update image weights
        # weights = list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs)))
        # weights = (imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX]*sqrt((1-bestError)/bestError) : weights[imgIDX]*sqrt(bestError/(1-bestError)), 1:numImgs)
        weights = map(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX]*sqrt((1-bestError)/bestError) : weights[imgIDX]*sqrt(bestError/(1-bestError)), 1:numImgs)

        # remove feature (a feature can't be selected twice)
        # feature_indexes.remove(best_feature_idx)
        featureIndices = filter!(e -> e ∉ bestFeatureIDX, featureIndices) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
        
        next!(p)
    end
    
    return classifiers
    
end


# function _get_feature_vote(feature::HaarLikeFeature, image::Int64)
#     # return getVote(image)
#     # return partial(getVote)(image)
#     # return feature.getVote(image)
#     return getVote(feature, image)
# end


function _create_features(imgHeight::Int64, imgWidth::Int64, minFeatureWidth::Int64, maxFeatureWidth::Int64, minFeatureHeight::Int64, maxFeatureHeight::Int64)
    println("Creating Haar-like features...")
    # features = Array()
    features = []
    
    # for feature in FeatureTypes # from HaarLikeFeature.jl
    #     # FeatureTypes are just tuples
    #     println(typeof(feature), " " , feature)
    # end # end for feature in feature types
    
    for feature in FeatureTypes # from HaarLikeFeature.jl
        # FeatureTypes are just tuples
        # println("feature: ", typeof(feature), " " , feature)
        featureStartWidth = max(minFeatureWidth, feature[1])
        for featureWidth in range(featureStartWidth, stop=maxFeatureWidth, step=feature[1])
            featureStartHeight = max(minFeatureHeight, feature[2])
            for featureHeight in range(featureStartHeight, stop=maxFeatureHeight, step=feature[2])
                for x in 1:(imgWidth - featureWidth)
                    for y in 1:(imgHeight - featureHeight)
                        # println("top left: ", typeof((x, y)), " ", (x, y))
                        features = push!(features, HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, 1))
                        # features = (features..., HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, 1)) # using splatting to add to a tuple.  see Utils.partial()
                        features = push!(features, HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, -1))
                        # features = (features..., HaarLikeFeature(feature, (x, y), featureWidth, featureHeight, 0, -1))
                        # feature.append(HaarLikeFeature...)
                    end # end for y
                end # end for x
            end # end for feature height
        end # end for feature width
    end # end for feature in feature types
    
    # println("...finished processing; ", length(features), " features created.")
    
    return features
end


export learn
export _get_feature_vote
export _create_features



### TESTING
