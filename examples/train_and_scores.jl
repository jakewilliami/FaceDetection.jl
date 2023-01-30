Threads.nthreads() > 1 || @warn(
    "You are currently only using one thread, when the programme supports multithreading"
)
@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest
using Serialization: deserialize

@info("...done")
function takerand!(list::Vector{T}) where {T}
    j = rand(1:length(list))
    rand_elem = list[j]
    deleteat!(list, j)
    return rand_elem
end

rand_subset!(list::Vector{T}, n::Int) where {T} = String[takerand!(list) for _ = 1:n]

"Return a random subset of the contents of directory `path` of size `n`."
function rand_subset_ls(path::String, n::Int)
    dir_contents = readdir(path, join = true, sort = false)
    filter!(f -> !occursin(r".*\.DS_Store", f), dir_contents)
    @assert(
        length(dir_contents) >= n,
        "Not enough files in given directory to select `n` random."
    )

    return rand_subset!(dir_contents, n)
end

function main(
    num_pos::Int,
    num_neg::Int;
    smart_choose_feats::Bool = false,
    scale::Bool = true,
    scale_to::Tuple = (128, 128),
)
    data_path = joinpath(dirname(@__DIR__), "data")

    pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
    neg_training_path = joinpath(data_path, "things", "object_images")

    all_pos_images = rand_subset_ls(pos_training_path, 2num_pos)
    all_neg_images = rand_subset_ls(neg_training_path, 2num_neg)

    pos_training_images = all_pos_images[1:num_pos]
    neg_training_images = all_neg_images[1:num_neg]

    num_classifiers = 10
    local min_size_img::Tuple{Int, Int}

    if smart_choose_feats
        # For performance reasons restricting feature size
        @info("Selecting best feature width and height...")

        max_feature_width,
        max_feature_height,
        min_feature_height,
        min_feature_width,
        min_size_img = determine_feature_size(
            vcat(pos_training_images, neg_training_images);
            scale = scale,
            scale_to = scale_to,
            show_progress = true,
        )

        @info(
            "...done.  Maximum feature width selected is $max_feature_width pixels; minimum feature width is $min_feature_width; maximum feature height is $max_feature_height pixels; minimum feature height is $min_feature_height.\n"
        )
    else
        # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (67, 67, 65, 65)
        # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (100, 100, 30, 30)
        max_feature_width, max_feature_height, min_feature_height, min_feature_width =
            (70, 70, 50, 50)
        min_size_img = (128, 128)
    end

    # classifiers are haar like features
    classifiers = learn(
        pos_training_images,
        neg_training_images,
        num_classifiers,
        min_feature_height,
        max_feature_height,
        min_feature_width,
        max_feature_width;
        scale = scale,
        scale_to = scale_to,
    )

    @info("Calculating test face scores and constructing dataset...")
    sleep(0.5)

    pos_testing_images = all_pos_images[(num_pos + 1):(2num_pos)]
    neg_testing_images = all_neg_images[(num_neg + 1):(2num_neg)]
    num_faces = length(pos_testing_images)
    num_non_faces = length(neg_testing_images)

    faces_scores = Vector{Real}(undef, num_faces)
    non_faces_scores = Vector{Real}(undef, num_non_faces)

    # faces_scores[:] .= (sum(get_faceness(c, load_image(face, scale=scale, scale_to=scale_to)) for c in classifiers) / num_classifiers for face in pos_testing_images)
    # non_faces_scores[:] .= (sum(get_faceness(c, load_image(non_face, scale=scale, scale_to=scale_to)) for c in classifiers) / num_classifiers for non_face in neg_testing_images)
    faces_scores[:] .= (
        get_faceness(classifiers, load_image(face, scale = scale, scale_to = scale_to)) for
        face in pos_testing_images
    )
    non_faces_scores[:] .= (
        get_faceness(classifiers, load_image(non_face, scale = scale, scale_to = scale_to)) for non_face in neg_testing_images
    )

    face_names = String[basename(i) for i in pos_testing_images]
    non_face_names = String[basename(i) for i in neg_testing_images]

    # filling in the dataset with missing to easily write to csv
    df_faces = faces_scores
    df_non_faces = non_faces_scores
    if length(faces_scores) < length(non_faces_scores)
        to_add = num_non_faces - num_faces
        df_faces = vcat(df_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        face_names = vcat(face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    elseif length(faces_scores) >= length(non_faces_scores)
        to_add = num_faces - num_non_faces
        df_non_faces = vcat(df_non_faces, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
        non_face_names =
            vcat(non_face_names, Matrix{Union{Float64, Missing}}(undef, to_add, 1))
    else
        error("unreachable")
    end

    # write score data
    data_file = joinpath(dirname(@__DIR__), "data", "faceness-scores.csv")
    write(
        data_file,
        DataFrame(hcat(face_names, df_faces, non_face_names, df_non_faces), :auto),
        writeheader = false,
    )

    @info("...done.  Dataset written to $(data_file).\n")
    @info("Computing differences in scores between faces and non-faces...")

    welch_t = UnequalVarianceTTest(faces_scores, non_faces_scores)

    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")

    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
        StatsPlots.boxplot(faces_scores, xaxis = false, label = false),
        StatsPlots.boxplot(non_faces_scores, xaxis = false, label = false),
        title = ["Scores of Faces" "Scores of Non-Faces"],
        # label = ["faces" "non-faces"],
        fontfamily = font("Times"),
        layout = @layout([a b]),
        # fillcolor = [:blue, :orange],
        link = :y,
        # framestyle = [:origin :origin]
    )

    # save plot
    StatsPlots.savefig(plot, joinpath(dirname(@__DIR__), "figs", "scores.pdf"))
    @info("...done.  Plot created at $(joinpath(dirname(@__DIR__), "figs", "scores.pdf"))")
end

@time main(200, 200, smart_choose_feats = false, scale = true, scale_to = (128, 128))
