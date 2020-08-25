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


function learn(positiveIIs::AbstractArray, negativeIIs::AbstractArray, numClassifiers::Int64=-1, minFeatureWidth::Int64=1, maxFeatureWidth::Int64=-1, minFeatureHeight::Int64=1, maxFeatureHeight::Int64=-1)
    #=
    The boosting algorithm for learning a query online.  $T$ hypotheses are constructed, each using a single feature.  The final hypothesis is a weighted linear combination of the $T$ hypotheses, where the weights are inverselt proportional to the training errors.
    This function selects a set of classifiers. Iteratively takes the best classifiers based on a weighted error.
    
    parameter `positiveIIs`: List of positive integral image examples [type: Abstracy Array]
    parameter `negativeIIs`: List of negative integral image examples [type: Abstract Array]
    parameter `numClassifiers`: Number of classifiers to select. -1 will use all
    classifiers [type: Integer]
    
    return `classifiers`: List of selected features [type: HaarLikeFeature]
    =#
    
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    numPos = length(positiveIIs)
    numNeg = length(negativeIIs)
    global numImgs = numPos + numNeg
    
    # get image height and width
    imgHeight, imgWidth = size(positiveIIs[1])
    
    # Maximum feature width and height default to image width and height
    # max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    # max_feature_width = img_width if max_feature_width == -1 else max_feature_width
    if isequal(maxFeatureHeight, -1)
        maxFeatureHeight = imgHeight
    end
    if isequal(maxFeatureWidth, -1)
        maxFeatureWidth = imgWidth
    end
    
    # Initialise weights $w_{1,i} = \frac{1}{2m}, \frac{1}{2l}$, for $y_i=0,1$ for negative and positive examples respectively
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
    
    # Concatenate positive and negative weights into one `weights` array
    weights = vcat(posWeights, negWeights)
    # println(weights)
    # weights = vcat(posWeights, negWeights)
    # weights = hcat((posWeights, negWeights))
    # weights = vcat([posWeights, negWeights])
    # labels = vcat((ones(numPos), ones(numNeg) * -1))
    labels = vcat(ones(numPos), ones(numNeg) * -1)
    # labels = vcat([ones(numPos), ones(numNeg) * -1])
    
    # get list of images (global because of @everywhere scope)
    # global images = positiveIIs + negativeIIs
    global images = vcat(positiveIIs, negativeIIs)

    # Create features for all sizes and locations
    global features = _create_features(imgHeight, imgWidth, minFeatureWidth, maxFeatureWidth, minFeatureHeight, maxFeatureHeight)
    numFeatures = length(features)
    # feature_indexes = list(range(num_features))
    featureIndices = Array(1:numFeatures)
    
    if isequal(numClassifiers, -1)
        numClassifiers = numFeatures
    end
    
    # println(typeof(numImgs));println(typeof(numFeatures))
    # create an empty array (of zeroes) with dimensions (numImgs, numFeautures)
    global votes = zeros((numImgs, numFeatures)) # necessarily different from `zero.((numImgs, numFeatures))`

    # bar = progressbar.ProgressBar()
    # @everywhere numImgs begin
    # println(votes)

    # show progress bar
    @everywhere begin
        n = numImgs
        processes = length(numImgs) # i.e., hypotheses
        p = Progress(n, 1)   # minimum update interval: 1 second
        for t in processes # bar(range(num_imgs)):
            # votes[i, :] = np.array(list(Pool(processes=None).map(partial(_get_feature_vote, image=images[i]), features)))
            # votes[i, :] = Array(map(partial(getVote, images[i]), features))
            votes[t, :] = Array(map(feature -> getVote(feature, images[t]), features))
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
    for t in processes
        # println(typeof(length(featureIndices)))
        classificationErrors = zeros(length(featureIndices))

        # normalize the weights $w_{t,i}\gets \frac{w_{t,i}}{\sum_{j=1}^n w_{t,j}}$
        # weights *= 1. / np.sum(weights)
        # weights *= float(1) / sum(weights)
        # weights = float(weights) / sum(weights) # ensure weights do not get rounded by element-wise making them floats
        # weights *= 1. / sum(weights)
        # weights = weights/sum(weights)
        # weights *= 1. / sum(weights)
        weights = deepdiv(deepfloat(weights), deepsum(weights))

        # For each feature j, train a classifier $h_j$ which is restricted to using a single feature.  The error is evaluated with respect to $w_j,\varepsilon_j = \sum_i w_i\left|h_j\left(x_i\right)-y_i\right|$
        for j in 1:length(featureIndices)
            fIDX = featureIndices[j]
            # classifier error is the sum of image weights where the classifier is right
            # error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            # error = sum(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, fIDX]) ? weights[imgIDX] : 0, 1:numImgs)
            # error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            ε = deepsum(map(imgIDX -> labels[imgIDX] ≠ votes[imgIDX, fIDX] ? weights[imgIDX] : 0, 1:numImgs))
            # lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0
            # imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, fIDX]) ? weights[imgIDX] : 0
            
            classificationErrors[j] = ε
        end

        # choose the classifier $h_t$ with the lowest error $\varepsilon_t$
        minErrorIDX = argmin(classificationErrors) # returns the index of the minimum in the array
        bestError = classificationErrors[minErrorIDX]
        bestFeatureIDX = featureIndices[minErrorIDX]

        # println(classificationErrors)
        # println(weights)
        # set feature weight
        bestFeature = features[bestFeatureIDX]
        # featureWeight = 0.5 * log((1 - bestError) / bestError)
        featureWeight = (1 - bestError) / bestError # β
        bestFeature.weight = featureWeight # need to element-wise alter the struct `weight`; else we get `setfield! immutable struct of type HaarLikeFeature cannot be changed`

        # classifiers = vcat(classifiers, bestFeature)
        classifiers = push!(classifiers, bestFeature)

        # update image weights $w_{t+1,i}=w_{t,i}\beta_{t}^{1-e_i}$
        # weights = list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs)))
        # weights = (imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX]*sqrt((1-bestError)/bestError) : weights[imgIDX]*sqrt(bestError/(1-bestError)), 1:numImgs)
        
        weights = map(imgIDX -> (labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]) ? weights[imgIDX] * featureWeight : weights[imgIDX] * featureWeight, 1:numImgs)
        
        # β = ε / (1 - ε)
        #
        # e = nothing
        #
        # if labels[imgIDX] ≠ votes[imgIDX, bestFeatureIDX]
        #     weights[imgIDX] * sqrt((1-bestError)/bestError)
        # else
        #     weights[imgIDX]*sqrt(bestError/(1-bestError)
        # end
        

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
    #=
    Iteratively creates the Haar-like feautures
    
    parameter `imgHeight`: The height of the image [type: Integer]
    parameter `imgWidth`: The width of the image [type: Integer]
    parameter `minFeatureWidth`: The minimum width of the feature (used for computation efficiency purposes) [type: Integer]
    parameter `maxFeatureWidth`: The maximum width of the feature [type: Integer]
    parameter `minFeatureHeight`: The minimum height of the feature [type: Integer]
    parameter `maxFeatureHeight`: The maximum height of the feature [type: Integer]
    
    return `features`: an array of Haar-like features found for an image [type: Abstract Array]
    =#
    
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
