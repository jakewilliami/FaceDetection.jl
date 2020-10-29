#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/../" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


# TODO: select optimal threshold for each feature
# TODO: attentional cascading

# include("HaarLikeFeature.jl")


using Base.Threads: @threads
using Base.Iterators: partition
using ProgressMeter: @showprogress, Progress, next!

function get_feature_votes(
    positive_path::AbstractString,
    negative_path::AbstractString,
    num_classifiers::Integer=Int32(-1),
    min_feature_width::Integer=Int32(1),
    max_feature_width::Integer=Int32(-1),
    min_feature_height::Integer=Int32(1),
    max_feature_height::Integer=Int32(-1);
    scale::Bool = false,
    scale_to::Tuple = (Int32(200), Int32(200))
    )

    #this transforms everything to maintain type stability
    s1 ,s2 = scale_to
    min_feature_width,
    max_feature_width,
    min_feature_height,
    max_feature_height,s1,s2 = promote(min_feature_width,
    max_feature_width,
    min_feature_height,
    max_feature_height,s1,s2)
    scale_to = (s1,s2)

    _Int = typeof(max_feature_width)
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    positive_files = filtered_ls(positive_path)
    negative_files = filtered_ls(negative_path)
    image_files = vcat(positive_files, negative_files)
    
    num_pos = length(positive_files)
    num_neg = length(negative_files)
    num_imgs = num_pos + num_neg
    
    # get image height and width
    temp_image = load_image(rand(positive_files), scale=scale, scale_to=scale_to)
    img_height, img_width = size(temp_image)
    temp_image = nothing # unload temporary image
    
    # Maximum feature width and height default to image width and height
    max_feature_height = isequal(max_feature_height, _Int(-1)) ? img_height : max_feature_height
    max_feature_width = isequal(max_feature_width, _Int(-1)) ? img_height : max_feature_width
    
    # Create features for all sizes and locations
    features = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = length(features)
    num_classifiers = isequal(num_classifiers, _Int(-1)) ? num_features : num_classifiers
    
    # create an empty array with dimensions (num_imgs, numFeautures)
    votes = Matrix{Int8}(undef, num_features, num_imgs)
    
    notify_user("Loading images ($(num_pos) positive and $(num_neg) negative images) and calculating their scores...")
    p = Progress(length(image_files), 1) # minimum update interval: 1 second
    num_processed = 0
    batch_size = 10
    # get votes for images
    map(partition(image_files, batch_size)) do batch
        ii_imgs = load_image.(batch; scale=scale, scale_to=scale_to)
        @threads for t in 1:length(batch)
            # votes[:, num_processed+t] .= get_vote.(features, Ref(ii_imgs[t]))
            map!(f -> get_vote(f, ii_imgs[t]), view(votes, :, num_processed + t), features)
            next!(p) # increment progress bar
        end
        num_processed += length(batch)
    end
    print("\n") # for a new line after the progress bar
    
    return votes, features
end

"""
    learn(
        positive_iis::AbstractArray,
        negative_iis::AbstractArray,
        num_classifiers::Int64=-1,
        min_feature_width::Int64=1,
        max_feature_width::Int64=-1,
        min_feature_height::Int64=1,
        max_feature_height::Int64=-1
    ) ->::Array{HaarLikeObject,1}

The boosting algorithm for learning a query online.  T hypotheses are constructed, each using a single feature.
The final hypothesis is a weighted linear combination of the T hypotheses, where the weights are inversely proportional to the training errors.
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
"""
function learn(
    positive_path::AbstractString,
    negative_path::AbstractString,
    features::AbstractArray,
    votes::AbstractArray,
    num_classifiers::Integer=-1
    )
    
    # get number of positive and negative images (and create a global variable of the total number of images——global for the @everywhere scope)
    num_pos = length(filtered_ls(positive_path))
    num_neg = length(filtered_ls(negative_path))
    num_imgs = num_pos + num_neg

    # Initialise weights $w_{1,i} = \frac{1}{2m}, \frac{1}{2l}$, for $y_i=0,1$ for negative and positive examples respectively
    pos_weights = ones(num_pos) / (2 * num_pos)
    neg_weights = ones(num_neg) / (2 * num_neg)

    # Concatenate positive and negative weights into one `weights` array
    weights = vcat(pos_weights, neg_weights)
    labels = vcat(ones(Int8, num_pos), ones(Int8, num_neg) * -one(Int8))
    
    num_features = length(features)

    feature_indices = Array(1:num_features)
    num_classifiers = isequal(num_classifiers, -1) ? num_features : num_classifiers
    
    notify_user("Selecting classifiers...")
    # select classifiers
    classifiers = []
    p = Progress(num_classifiers, 1) # minimum update interval: 1 second
    for t in 1:num_classifiers
        # classification_errors = zeros(length(feature_indices))
        classification_errors = Matrix{Float64}(undef, length(feature_indices), 1)
        # normalize the weights $w_{t,i}\gets \frac{w_{t,i}}{\sum_{j=1}^n w_{t,j}}$
        weights = float(weights) / sum(weights)
        # For each feature j, train a classifier $h_j$ which is restricted to using a single feature.  The error is evaluated with respect to $w_j,\varepsilon_j = \sum_i w_i\left|h_j\left(x_i\right)-y_i\right|$
        map!(view(classification_errors, :), 1:length(feature_indices)) do j
            sum(1:num_imgs) do img_idx
                labels[img_idx] ≠ votes[feature_indices[j], img_idx] ? weights[img_idx] : zero(Float64)
            end
        end

        # choose the classifier $h_t$ with the lowest error $\varepsilon_t$
        best_error, min_error_idx = findmin(classification_errors)
        best_feature_idx = feature_indices[min_error_idx]

        # set feature weight
        best_feature = features[best_feature_idx]
        feature_weight = β(best_error) # β
        best_feature.weight = feature_weight

        classifiers = push!(classifiers, best_feature)

        sqrt_best_error = @fastmath(sqrt(best_error / (one(best_error) - best_error)))

        # update image weights $w_{t+1,i}=w_{t,i}\beta_{t}^{1-e_i}$
        weights = map(i -> labels[i] ≠ votes[best_feature_idx, i] ? weights[i] * sqrt_best_error : weights[i] * sqrt_best_error, 1:num_imgs)

        # remove feature (a feature can't be selected twice)
        filter!(e -> e ∉ best_feature_idx, feature_indices) # note: without unicode operators, `e ∉ [a, b]` is `!(e in [a, b])`
        
        next!(p) # increment progress bar
    end
    
    print("\n") # for a new line after the progress bar
    
    return classifiers
    
end

function β(err::T)::T where T
    _1=one(err)
    _half = T(0.5)
    @fastmath(_half*log((_1 - err) / err))
end

function learn(
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
    
    votes, features = get_feature_votes(
        positive_path,
        negative_path,
        num_classifiers,
        min_feature_width,
        max_feature_width,
        min_feature_height,
        max_feature_height,
        scale = scale,
        scale_to = scale_to
    )
    
    return learn(positive_path, negative_path, features, votes, num_classifiers)
end

"""
    create_features(
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
"""
function create_features(
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
                        push!(features, HaarLikeObject(feature, (x, y), feature_width, feature_height, 0, 1))
                        push!(features, HaarLikeObject(feature, (x, y), feature_width, feature_height, 0, -1))
                    end # end for y
                end # end for x
            end # end for feature height
        end # end for feature width
    end # end for feature in feature types
    
    println("...finished processing; ", length(features), " features created.\n")
    
    return features
end
