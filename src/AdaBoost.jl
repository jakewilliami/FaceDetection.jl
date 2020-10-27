#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


# TODO: select optimal threshold for each feature
# TODO: attentional cascading

# include("HaarLikeFeature.jl")
include("Utils.jl")
include("IntegralImage.jl")

using ProgressMeter: @showprogress, Progress, next!
using SparseArrays: spzeros

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
    # positive_iis::AbstractArray,
    # negative_iis::AbstractArray,
    positive_path::AbstractString,
    negative_path::AbstractString,
    num_classifiers::Integer=-1,
    min_feature_width::Integer=1,
    max_feature_width::Integer=-1,
    min_feature_height::Integer=1,
    max_feature_height::Integer=-1;
    scale::Bool = false,
    scale_to::Tuple = (200, 200)
)::Array{HaarLikeObject,1}
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    
    positive_files = filtered_ls(positive_path)
    negative_files = filtered_ls(negative_path)
    
    num_pos = length(positive_files)
    num_neg = length(negative_files)
    num_imgs = num_pos + num_neg
    
    # get image height and width
    # temp_image = convert(Array{Float64}, Gray.(load(rand(positive_files))))
    temp_image = load_image(rand(positive_files), scale=scale, scale_to=scale_to)
    img_height, img_width = size(temp_image)
    temp_image = nothing # unload temporary image
    
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
    
    # Create features for all sizes and locations
    features = _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = length(features)
    feature_indices = Array(1:num_features)
    used = []
    if isequal(num_classifiers, -1)
        num_classifiers = num_features
    end
    
    # create an empty array (of zeroes) with dimensions (num_imgs, numFeautures)
    # votes = zeros((num_imgs, num_features)) # necessarily different from `zero.((num_imgs, num_features))`; previously zerosarray
    votes = zeros(num_imgs, num_features)
    num_processed = 0
    
    notify_user("Loading images ($(num_pos) positive and $(num_neg) negative images) and calculating their scores...")
    image_files = vcat(positive_files, negative_files)
    
    # instead of @showprogress, need to manually create the progress bar
    p = Progress(length(image_files), 1)
    # get votes for images
    # n = 10
    # map(Base.Iterators.partition(image_files, n)) do image_file
    #     ii_imgs = load_image.(image_file, scale=scale, scale_to=scale_to)
    #     for t in 1:n
    #         map!(f -> get_vote(f, ii_imgs[t]), view(votes, num_processed + t, :), features)
    #         num_processed += 1
    #     end
    #
    #     # increment progress bar
    #     next!(p)
    # end
    Base.Threads.@threads for image_file in image_files
        ii_img = load_image(image_file, scale=scale, scale_to=scale_to)
        num_processed += 1
        # votes[num_processed, :] .= map(f -> get_vote(f, ii_img), features)
        map!(f -> get_vote(f, ii_img), view(votes, num_processed, :), features)
    
        # increment progress bar
        next!(p)
    end # end loop through images
    print("\n") # for a new line after the progress bar
    
    notify_user("Selecting classifiers...")
    # select classifiers
    classifiers = []
    p = Progress(num_classifiers, 1)
    Base.Threads.@threads for t in 1:num_classifiers # previously, zerosarray
        classification_errors = zeros(length(feature_indices))
        # normalize the weights $w_{t,i}\gets \frac{w_{t,i}}{\sum_{j=1}^n w_{t,j}}$
        weights = float(weights) / sum(weights)

        # For each feature j, train a classifier $h_j$ which is restricted to using a single feature.  The error is evaluated with respect to $w_j,\varepsilon_j = \sum_i w_i\left|h_j\left(x_i\right)-y_i\right|$
        for j in 1:length(feature_indices)
            f_idx = feature_indices[j]
            # classifier error is the sum of image weights where the classifier is right
            # ε = sum(map(img_idx -> labels[img_idx] ≠ votes[img_idx, f_idx] ? weights[img_idx] : zero(Integer), 1:num_imgs))
            ε = sum([labels[img_idx] ≠ votes[img_idx, f_idx] ? weights[img_idx] : zero(Integer) for img_idx in 1:num_imgs])
            
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
        # weights = Array(map(i -> labels[i] ≠ votes[i, best_feature_idx] ? weights[i] * sqrt((1 - best_error) / best_error) : weights[i] * sqrt(best_error / (1 - best_error)), 1:num_imgs))
        weights = [labels[i] ≠ votes[i, best_feature_idx] ? weights[i] * sqrt((1 - best_error) / best_error) : weights[i] * sqrt(best_error / (1 - best_error)) for i in 1:num_imgs]

        # remove feature (a feature can't be selected twice)
        feature_indices = filter!(e -> e ∉ best_feature_idx, feature_indices) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
        
        # increment progress bar
        next!(p)
    end
    
    print("\n") # for a new line after the progress bar
    
    return classifiers
end

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
    
    for feature in values(feature_types) # (feature_types are just tuples)
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
