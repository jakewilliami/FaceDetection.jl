#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


# TODO: select optimal threshold for each feature
# TODO: attentional cascading


module AdaBoost

include("HaarLikeFeature.jl")
include("Utils.jl")

using ProgressMeter: @showprogress
using .HaarLikeFeature: FeatureTypes, HaarLikeObject, getVote
using .Utils: notifyUser

export learn, _create_features


function learn(positiveIIs::AbstractArray, negativeIIs::AbstractArray, numClassifiers::Int64=-1, minFeatureWidth::Int64=1, maxFeatureWidth::Int64=-1, minFeatureHeight::Int64=1, maxFeatureHeight::Int64=-1)#::Array{HaarLikeObject,1}
    #=
    The boosting algorithm for learning a query online.  $T$ hypotheses are constructed, each using a single feature.  The final hypothesis is a weighted linear combination of the $T$ hypotheses, where the weights are inversely proportional to the training errors.
    This function selects a set of classifiers. Iteratively takes the best classifiers based on a weighted error.
    
    parameter `positiveIIs`: List of positive integral image examples [type: Abstracy Array]
    parameter `negativeIIs`: List of negative integral image examples [type: Abstract Array]
    parameter `numClassifiers`: Number of classifiers to select. -1 will use all
    classifiers [type: Integer]
    
    return `classifiers`: List of selected features [type: HaarLikeObject]
    =#
    
    
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    numPos = length(positiveIIs)
    numNeg = length(negativeIIs)
    numImgs = numPos + numNeg
    
    # get image height and width
    imgHeight, imgWidth = size(positiveIIs[1])
    
    # Maximum feature width and height default to image width and height
    if isequal(maxFeatureHeight, -1)
        maxFeatureHeight = imgHeight
    end
    if isequal(maxFeatureWidth, -1)
        maxFeatureWidth = imgWidth
    end
    
    # Initialise weights $w_{1,i} = \frac{1}{2m}, \frac{1}{2l}$, for $y_i=0,1$ for negative and positive examples respectively
    posWeights = float(ones(numPos)) / (2 * numPos)
    negWeights = float(ones(numNeg)) / (2 * numNeg)
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
    =#
    
    # Concatenate positive and negative weights into one `weights` array
    weights = vcat(posWeights, negWeights)
    labels = vcat(ones(numPos), ones(numNeg) * -1)
    
    # get list of images (global because of @everywhere scope)
    images = vcat(positiveIIs, negativeIIs)

    # Create features for all sizes and locations
    features = _create_features(imgHeight, imgWidth, minFeatureWidth, maxFeatureWidth, minFeatureHeight, maxFeatureHeight)
    numFeatures = length(features)
    featureIndices = Array(1:numFeatures)
    used = []
    
    if isequal(numClassifiers, -1)
        numClassifiers = numFeatures
    end
    
    Utils.notifyUser("Calculating scores for images...")
    
    # create an empty array (of zeroes) with dimensions (numImgs, numFeautures)
    votes = zeros((numImgs, numFeatures)) # necessarily different from `zero.((numImgs, numFeatures))`; previously zerosarray

    n = numImgs
    processes = numImgs # i.e., hypotheses
    @showprogress for t in 1:processes # bar(range(num_imgs)):
        votes[t, :] = Array(map(f -> getVote(f, images[t]), features))
    end # end show progress in for loop
    
    print("\n") # for a new line after the progress bar
    
    # select classifiers
    classifiers = []

    Utils.notifyUser("Selecting classifiers...")
    
    n = numClassifiers
    @showprogress for t in 1:numClassifiers
        classificationErrors = zeros(length(featureIndices)) # previously, zerosarray

        # normalize the weights $w_{t,i}\gets \frac{w_{t,i}}{\sum_{j=1}^n w_{t,j}}$
        weights = float(weights) / sum(weights)

        # For each feature j, train a classifier $h_j$ which is restricted to using a single feature.  The error is evaluated with respect to $w_j,\varepsilon_j = \sum_i w_i\left|h_j\left(x_i\right)-y_i\right|$
        for j in 1:length(featureIndices)
            fIDX = featureIndices[j]
            # classifier error is the sum of image weights where the classifier is right
            ε = sum(map(imgIDX -> labels[imgIDX] ≠ votes[imgIDX, fIDX] ? weights[imgIDX] : 0, 1:numImgs))
            
            classificationErrors[j] = ε
        end

        # choose the classifier $h_t$ with the lowest error $\varepsilon_t$
        minErrorIDX = argmin(classificationErrors) # returns the index of the minimum in the array # consider `findmin`
        bestError = classificationErrors[minErrorIDX]
        bestFeatureIDX = featureIndices[minErrorIDX]

        # set feature weight
        bestFeature = features[bestFeatureIDX]
        featureWeight = 0.5 * log((1 - bestError) / bestError) # β
        bestFeature.weight = featureWeight

        classifiers = push!(classifiers, bestFeature)

        # update image weights $w_{t+1,i}=w_{t,i}\beta_{t}^{1-e_i}$
        weights = Array(map(i -> labels[i] ≠ votes[i, bestFeatureIDX] ? weights[i] * sqrt((1 - bestError) / bestError) : weights[i] * sqrt(bestError / (1 - bestError)), 1:numImgs))

        # remove feature (a feature can't be selected twice)
        featureIndices = filter!(e -> e ∉ bestFeatureIDX, featureIndices) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
    end
    
    print("\n") # for a new line after the progress bar
    
    return classifiers
    
end


#find / update threshold and coeff for each feature
# function _feature_job(feature_nr, feature)
#     #    if (feature_nr+1) % 1000 == 0:
#     #        print('[ %d of %d ]'%(feature_nr+1, n_features))
#     if feature_nr ∈ used
#         return
#     end
#
#     # find the scores for the images
#     scores = zeros(n_img)
#     for i, img in enumerate(images):
#         scores[i] = feature.get_score(img)
#     sorted_img_args = np.argsort(scores)
#     Sp = np.zeros(n_img)  # sum weights for positive examples below current img
#     Sn = np.zeros(n_img)  # sum weights for negative examples below current img
#     Tp = 0
#     Tn = 0
#     for img_arg in np.nditer(sorted_img_args):
#         if labels[img_arg] == 0:
#             Tn += w[img_arg]
#             Sn[img_arg] = Tn
#         else:
#             Tp += w[img_arg]
#             Sp[img_arg] = Tp
#
#     # compute the formula for the threshold
#     nerror = Sp + (Tn - Sn)  # error of classifying everything negative below threshold
#     perror = Sn + (Tp - Sp)  # error of classifying everything positive below threshold
#     error = np.minimum(perror, nerror)  # find minimum
#     best_threshold_img = np.argmin(error)  # find the image with the threshold
#     best_local_error = error[best_threshold_img]
#     feature.threshold = scores[best_threshold_img]  # use the score we estimated for the image as new threshold
#     # assign new polarity, based on above calculations
#     feature.polarity = 1 if nerror[best_threshold_img] < perror[best_threshold_img] else -1
#
#     # store the error to find best feature
#     errors[feature_nr] = best_local_error
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
    
    Utils.notifyUser("Creating Haar-like features...")
    features = []
    
    if imgWidth < maxFeatureWidth || imgHeight < maxFeatureHeight
        error("""
        Cannot possibly find classifiers whose size is greater than the image itself [(width,height) = ($imgWidth,$imgHeight)].
        """)
    end
    
    for feature in HaarLikeFeature.FeatureTypes # (FeatureTypes are just tuples)
        featureStartWidth = max(minFeatureWidth, feature[1])
        for featureWidth in range(featureStartWidth, stop=maxFeatureWidth, step=feature[1])
            featureStartHeight = max(minFeatureHeight, feature[2])
            for featureHeight in range(featureStartHeight, stop=maxFeatureHeight, step=feature[2])
                for x in 1:(imgWidth - featureWidth)
                    for y in 1:(imgHeight - featureHeight)
                        features = push!(features, HaarLikeFeature.HaarLikeObject(feature, (x, y), featureWidth, featureHeight, 0, 1))
                        features = push!(features, HaarLikeFeature.HaarLikeObject(feature, (x, y), featureWidth, featureHeight, 0, -1))
                        # features = push!(features, HaarLikeFeature.HaarLikeObject(feature, (x, y), featureWidth, featureHeight, 0.11, -1))
                    end # end for y
                end # end for x
            end # end for feature height
        end # end for feature width
    end # end for feature in feature types
    
    println("...finished processing; ", length(features), " features created.\n")
    
    return features
end


end # end module
