#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


# TODO: select optimal threshold for each feature
# TODO: attentional cascading

# include("HaarLikeFeature.jl")
include("Utils.jl")

using ProgressMeter: @showprogress
#=
    learn(
        positive_iis::AbstractArray,
        negative_iis::AbstractArray,
        num_classifiers::Int64=-1,
        min_feature_width::Int64=1,
        max_feature_width::Int64=-1,
        min_feature_height::Int64=1,
        max_feature_height::Int64=-1
    ) ->::Array{HaarLikeObject,1}

The boosting algorithm for learning a query online.  $T$ hypotheses are constructed, each using a single feature.
The final hypothesis is a weighted linear combination of the $T$ hypotheses, where the weights are inversely proportional to the training errors.
This function selects a set of classifiers. Iteratively takes the best classifiers based on a weighted error.

# Arguments

- `positive_iis::AbstractArray`: List of positive integral image examples
- `negative_iis::AbstractArray`: List of negative integral image examples
- `num_classifiers::Integer`: Number of classifiers to select. -1 will use all classifiers
- `min_feature_width::Integer`: the minimum width of the feature
- `max_feature_width::Integer`: the maximum width of the feature
- `min_feature_height::Integer`: the minimum height of the feature
- `max_feature_width::Integer`: the maximum height of the feature

# Returns `classifiers::Array{HaarLikeObject, 1}`: List of selected features
=#
function learn(
    positive_iis::AbstractArray,
    negative_iis::AbstractArray,
    num_classifiers::Integer=-1,
    min_feature_width::Integer=1,
    max_feature_width::Integer=-1,
    min_feature_height::Integer=1,
    max_feature_height::Integer=-1
)#::Array{HaarLikeObject,1}
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    num_pos = length(positive_iis)
    num_neg = length(negative_iis)
    num_imgs = num_pos + num_neg
    
    # get image height and width
    img_height, img_width = size(positive_iis[1])
    
    # Maximum feature width and height default to image width and height
    if isequal(max_feature_height, -1)
        max_feature_height = img_height
    end
    if isequal(max_feature_width, -1)
        max_feature_width = img_width
    end
    
    # Initialise weights $w_{1,i} = \frac{1}{2m}, \frac{1}{2l}$, for $y_i=0,1$ for negative and positive examples respectively
    pos_weights = float(ones(num_pos)) / (2 * num_pos)
    neg_weights = float(ones(num_neg)) / (2 * num_neg)
    #=
    Consider the original code,
        ```
        $ python3 -c 'import numpy; a=[1,2,3]; b=[4,5,6]; print(numpy.hstack((a,b)))'
        [1 2 3 4 5 6]
        ```
    This is *not* comparable to Julia's `hcat`.  Here,
        ```
        numpy.hstack((a,b)) ≡ vcat(a,b)
        ```
    =#
    
    # Concatenate positive and negative weights into one `weights` array
    weights = vcat(pos_weights, neg_weights)
    labels = vcat(ones(num_pos), ones(num_neg) * -1)
    
    # get list of images (global because of @everywhere scope)
    images = vcat(positive_iis, negative_iis)

    # Create features for all sizes and locations
    features = _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = length(features)
    feature_indices = Array(1:num_features)
    used = []
    
    if isequal(num_classifiers, -1)
        num_classifiers = num_features
    end
    
    notify_user("Calculating scores for images...")
    
    # create an empty array (of zeroes) with dimensions (num_imgs, numFeautures)
    votes = zeros((num_imgs, num_features)) # necessarily different from `zero.((num_imgs, num_features))`; previously zerosarray

    n = num_imgs
    processes = num_imgs # i.e., hypotheses
    @showprogress for t in 1:processes # bar(range(num_imgs)):
        votes[t, :] = Array(map(f -> get_vote(f, images[t]), features))
    end # end show progress in for loop
    
    print("\n") # for a new line after the progress bar
    
    # select classifiers
    classifiers = []

    notify_user("Selecting classifiers...")
    
    n = num_classifiers
    @showprogress for t in 1:num_classifiers
        classification_errors = zeros(length(feature_indices)) # previously, zerosarray

        # normalize the weights $w_{t,i}\gets \frac{w_{t,i}}{\sum_{j=1}^n w_{t,j}}$
        weights = float(weights) / sum(weights)

        # For each feature j, train a classifier $h_j$ which is restricted to using a single feature.  The error is evaluated with respect to $w_j,\varepsilon_j = \sum_i w_i\left|h_j\left(x_i\right)-y_i\right|$
        for j in 1:length(feature_indices)
            f_idx = feature_indices[j]
            # classifier error is the sum of image weights where the classifier is right
            ε = sum(map(img_idx -> labels[img_idx] ≠ votes[img_idx, f_idx] ? weights[img_idx] : 0, 1:num_imgs))
            
            classification_errors[j] = ε
        end

        # choose the classifier $h_t$ with the lowest error $\varepsilon_t$
        min_error_idx = argmin(classification_errors) # returns the index of the minimum in the array # consider `findmin`
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indices[min_error_idx]

        # set feature weight
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * log((1 - best_error) / best_error) # β
        best_feature.weight = feature_weight

        classifiers = push!(classifiers, best_feature)

        # update image weights $w_{t+1,i}=w_{t,i}\beta_{t}^{1-e_i}$
        weights = Array(map(i -> labels[i] ≠ votes[i, best_feature_idx] ? weights[i] * sqrt((1 - best_error) / best_error) : weights[i] * sqrt(best_error / (1 - best_error)), 1:num_imgs))

        # remove feature (a feature can't be selected twice)
        feature_indices = filter!(e -> e ∉ best_feature_idx, feature_indices) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
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

#=
    _create_features(
        img_height::Integer,
        img_width::Integer,
        min_feature_width::Integer,
        max_feature_width::Integer,
        min_feature_height::Integer,
        max_feature_height::Integer
    ) -> Array{HaarLikeObject, 1}

Iteratively creates the Haar-like feautures

# Arguments

- `img_height::Integer`: The height of the image
- `img_width::Integer`: The width of the image
- `min_feature_width::Integer`: The minimum width of the feature (used for computation efficiency purposes)
- `max_feature_width::Integer`: The maximum width of the feature
- `min_feature_height::Integer`: The minimum height of the feature
- `max_feature_height::Integer`: The maximum height of the feature

# Returns

- `features::AbstractArray`: an array of Haar-like features found for an image
=#
function _create_features(
    img_height::Integer,
    img_width::Integer,
    min_feature_width::Integer,
    max_feature_width::Integer,
    min_feature_height::Integer,
    max_feature_height::Integer
)
    notify_user("Creating Haar-like features...")
    features = []
    
    if img_width < max_feature_width || img_height < max_feature_height
        error("""
        Cannot possibly find classifiers whose size is greater than the image itself [(width,height) = ($img_width,$img_height)].
        """)
    end
    
    for feature in feature_types # (feature_types are just tuples)
        feature_start_width = max(min_feature_width, feature[1])
        for feature_width in range(feature_start_width, stop=max_feature_width, step=feature[1])
            feature_start_height = max(min_feature_height, feature[2])
            for feature_height in range(feature_start_height, stop=max_feature_height, step=feature[2])
                for x in 1:(img_width - feature_width)
                    for y in 1:(img_height - feature_height)
                        features = push!(features, HaarLikeObject(feature, (x, y), feature_width, feature_height, 0, 1))
                        features = push!(features, HaarLikeObject(feature, (x, y), feature_width, feature_height, 0, -1))
                    end # end for y
                end # end for x
            end # end for feature height
        end # end for feature width
    end # end for feature in feature types
    
    println("...finished processing; ", length(features), " features created.\n")
    
    return features
end
