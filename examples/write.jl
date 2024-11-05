# println("\033[1;34m===>\033[0;38m\033[1;38m\tLoading required libraries (it will take a moment to precompile if it is your first time doing this)...\033[0;38m")
@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

using FaceDetection
const FD = FaceDetection
using Printf: @printf
using Serialization: serialize

@info("...done\n")

function main(;
    smart_choose_feats::Bool = false, scale::Bool = false, scale_to::Tuple = (200, 200)
)
    include("constants.jl")
    include("main_data.jl")

    min_size_img = (19, 19) # default for our test dataset
    if smart_choose_feats
        # For performance reasons restricting feature size
        @info("Selecting best feature width and height...")

        max_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = determine_feature_size(
            pos_training_path, neg_training_path; scale = scale, scale_to = scale_to
        )

        @info("...done.  Maximum feature width selected is $max_feature_width pixels; minimum feature width is $min_feature_width; maximum feature height is $max_feature_height pixels; minimum feature height is $min_feature_height.\n")
    else
        min_feature_height = 8
        max_feature_height = 10
        min_feature_width = 8
        max_feature_width = 10
    end

    # classifiers are haar like features
    votes = FD.get_feature_votes(
        pos_training_path,
        neg_training_path,
        num_classifiers,
        min_feature_height,
        max_feature_height,
        min_feature_width,
        max_feature_width;
        scale = scale,
        scale_to = scale_to,
    )

    # write classifiers to file
    img_size = scale ? scale_to : min_size_img
    data_file = joinpath(@__DIR__, "data", "feature_votes_$(img_size)")
    return serialize(data_file, votes)
end

@time main(smart_choose_feats = true, scale = true, scale_to = (20, 20))
