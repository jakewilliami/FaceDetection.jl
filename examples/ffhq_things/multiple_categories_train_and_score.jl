Threads.nthreads() > 1 ||
    @warn("You are currently only using one thread, when the programme supports multithreading")
@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

using FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest
using Serialization: deserialize

@enum ImageType begin
    FACE = 1
    PAREIDOLIA = 2
    FLOWER = 3
    OBJECT = 4
end

@info("...done")
function takerand!(list::Vector{T}) where {T}
    j = rand(1:length(list))
    rand_elem = list[j]
    deleteat!(list, j)
    return rand_elem
end

rand_subset!(list::Vector{T}, n::Int) where {T} = String[takerand!(list) for _ in 1:n]

"Return a random subset of the contents of directory `path` of size `n`."
function rand_subset_ls(path::String, n::Int)
    dir_contents = readdir(path; join = true, sort = false)
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
    data_path = joinpath(dirname(dirname(@__DIR__)), "data")

    pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
    neg_training_path = joinpath(data_path, "things", "object_images")
    testing_path = joinpath(data_path, "lizzie-testset", "2021")

    pos_training_images = rand_subset_ls(pos_training_path, num_pos)
    neg_training_images = rand_subset_ls(neg_training_path, num_neg)

    num_classifiers = 10
    local min_size_img::Tuple{Int, Int}

    if smart_choose_feats
        # For performance reasons restricting feature size
        @info("Selecting best feature width and height...")

        max_feature_width, max_feature_height, min_feature_height, min_feature_width, min_size_img = determine_feature_size(
            vcat(pos_training_images, neg_training_images);
            scale = scale,
            scale_to = scale_to,
            show_progress = true,
        )

        @info("...done.  Maximum feature width selected is $max_feature_width pixels; minimum feature width is $min_feature_width; maximum feature height is $max_feature_height pixels; minimum feature height is $min_feature_height.\n")
    else
        # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (67, 67, 65, 65)
        max_feature_width, max_feature_height, min_feature_height, min_feature_width = (
            100, 100, 30, 30
        )
        # max_feature_width, max_feature_height, min_feature_height, min_feature_width = (70, 70, 50, 50)
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
    println(classifiers)
    @info("Calculating test face scores and constructing dataset...")
    sleep(0.5)

    face_testing_images = readdir(
        joinpath(testing_path, "Faces"); join = true, sort = false
    )
    pareidolia_testing_images = readdir(
        joinpath(testing_path, "Pareidolia"); join = true, sort = false
    )
    flower_testing_images = readdir(
        joinpath(testing_path, "Flowers"); join = true, sort = false
    )
    object_testing_images = readdir(
        joinpath(testing_path, "Objects"); join = true, sort = false
    )

    num_faces = length(face_testing_images)
    num_pareidolia = length(pareidolia_testing_images)
    num_flowers = length(flower_testing_images)
    num_objects = length(object_testing_images)

    face_scores = Float64[
        get_faceness(classifiers, load_image(img; scale = scale, scale_to = scale_to)) for
        img in face_testing_images
    ]
    pareidolia_scores = Float64[
        get_faceness(classifiers, load_image(img; scale = scale, scale_to = scale_to)) for
        img in pareidolia_testing_images
    ]
    flower_scores = Float64[
        get_faceness(classifiers, load_image(img; scale = scale, scale_to = scale_to)) for
        img in flower_testing_images
    ]
    object_scores = Float64[
        get_faceness(classifiers, load_image(img; scale = scale, scale_to = scale_to)) for
        img in object_testing_images
    ]

    face_names = String[basename(img) for img in face_testing_images]
    pareidolia_names = String[basename(img) for img in pareidolia_testing_images]
    flower_names = String[basename(img) for img in flower_testing_images]
    object_names = String[basename(img) for img in object_testing_images]

    scores_df = DataFrame(; image_name = String[], image_score = Float64[])
    anova_df = DataFrame(; image_type = Int[], image_score = Float64[])
    for (image_type, image_names, image_scores) in (
        (FACE, face_names, face_scores),
        (PAREIDOLIA, pareidolia_names, pareidolia_scores),
        (FLOWER, flower_names, flower_scores),
        (OBJECT, object_names, object_scores),
    )
        for (n, s) in zip(image_names, image_scores)
            push!(scores_df, (n, s))
            push!(anova_df, (Int(image_type), s))
        end
    end

    # write score data
    data_file = joinpath(dirname(dirname(@__DIR__)), "data", "faceness-scores.csv")
    write(data_file, scores_df; writeheader = false)
    @info("...done.  Dataset written to $(data_file).\n")

    ### FACES VS OBJECTS

    @info("Computing differences in scores between faces and objects...")
    welch_t = UnequalVarianceTTest(face_scores, object_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")

    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
        StatsPlots.boxplot(face_scores; xaxis = false, label = false),
        StatsPlots.boxplot(object_scores; xaxis = false, label = false);
        title = ["Facenesses of Faces" "Facenesses of Objects"],
        fontfamily = font("Times"),
        layout = @layout([a b]),
        link = :y,
    )
    StatsPlots.savefig(
        plot,
        joinpath(
            dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_objects.pdf"
        ),
    )
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_objects.pdf"))")

    ### PAREIDOLIA VS OBJECTS

    @info("Computing differences in scores between pareidolia and objects...")
    welch_t = UnequalVarianceTTest(pareidolia_scores, object_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")

    plot = StatsPlots.plot(
        StatsPlots.boxplot(pareidolia_scores; xaxis = false, label = false),
        StatsPlots.boxplot(object_scores; xaxis = false, label = false);
        title = ["Facenesses of Pareidolia" "Facenesses of Objects"],
        fontfamily = font("Times"),
        layout = @layout([a b]),
        link = :y,
    )
    StatsPlots.savefig(
        plot,
        joinpath(
            dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_objects.pdf"
        ),
    )
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_objects.pdf"))")

    ### FACES VS FLOWERS

    @info("Computing differences in scores between faces and flowers...")
    welch_t = UnequalVarianceTTest(face_scores, flower_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")

    plot = StatsPlots.plot(
        StatsPlots.boxplot(face_scores; xaxis = false, label = false),
        StatsPlots.boxplot(flower_scores; xaxis = false, label = false);
        title = ["Facenesses of Faces" "Facenesses of Flowers"],
        fontfamily = font("Times"),
        layout = @layout([a b]),
        link = :y,
    )
    StatsPlots.savefig(
        plot,
        joinpath(
            dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_flowers.pdf"
        ),
    )
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_flowers.pdf"))")

    ### PAREIDOLIA VS FLOWERS

    @info("Computing differences in scores between pareidolia and flowers...")
    welch_t = UnequalVarianceTTest(pareidolia_scores, flower_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")

    plot = StatsPlots.plot(
        StatsPlots.boxplot(pareidolia_scores; xaxis = false, label = false),
        StatsPlots.boxplot(flower_scores; xaxis = false, label = false);
        title = ["Facenesses of Pareidolia" "Facenesses of Flowers"],
        fontfamily = font("Times"),
        layout = @layout([a b]),
        link = :y,
    )
    StatsPlots.savefig(
        plot,
        joinpath(
            dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_flowers.pdf"
        ),
    )
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_flowers.pdf"))")
end

@time main(500, 500, smart_choose_feats = false, scale = true, scale_to = (128, 128))
