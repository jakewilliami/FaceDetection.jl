Threads.nthreads() > 1 || @warn("You are currently only using one thread, when the programme supports multithreading")
@info "Loading required libraries (it will take a moment to precompile if it is your first time doing this)..."

using Serialization: deserialize

include(joinpath(dirname(dirname(@__DIR__)), "src", "FaceDetection.jl"))

using .FaceDetection
using Images: imresize
using StatsPlots  # StatsPlots required for box plots # plot boxplot @layout :origin savefig
using CSV: write
using DataFrames: DataFrame
using HypothesisTests: UnequalVarianceTTest
using Serialization: deserialize

@enum ImageType begin
    FACE       = 1
    PAREIDOLIA = 2
    FLOWER     = 3
    OBJECT     = 4
end

@info("...done")
function takerand!(list::Vector{T}) where {T}
    j = rand(1:length(list))
    rand_elem = list[j]
    deleteat!(list, j)
    return rand_elem
end

rand_subset!(list::Vector{T}, n::Int) where {T} = 
    String[takerand!(list) for _ in 1:n]

"Return a random subset of the contents of directory `path` of size `n`."
function rand_subset_ls(path::String, n::Int)
	dir_contents = readdir(path, join=true, sort=false)
	filter!(f -> !occursin(r".*\.DS_Store", f), dir_contents)
	@assert(length(dir_contents) >= n, "Not enough files in given directory to select `n` random.")
	
    return rand_subset!(dir_contents, n)
end

function main(
    classifiers_file::String,
	num_pos::Int,
	num_neg::Int;
	scale::Bool=true,
	scale_to::Tuple=(128, 128)
)
   	classifiers = deserialize(classifiers_file)
    sort!(classifiers, by = c -> c.weight, rev = true)
    println(classifiers)
    
    @info("Calculating test face scores and constructing dataset...")
    sleep(0.5)
    
    data_path = joinpath(dirname(dirname(@__DIR__)), "data")
	
	pos_training_path = joinpath(data_path, "ffhq", "thumbnails128x128")
	neg_training_path = joinpath(data_path, "things", "object_images")
    testing_path = joinpath(data_path, "lizzie-testset", "2021")
    
    #=
    correct_faces = sum(ensemble_vote_all(pos_testing_images, classifiers, scale=scale, scale_to=scale_to))
	correct_non_faces = num_non_faces - sum(ensemble_vote_all(neg_testing_images, classifiers, scale=scale, scale_to=scale_to))
	correct_faces_percent = (correct_faces / num_faces) * 100
	correct_non_faces_percent = (correct_non_faces / num_non_faces) * 100

    faces_frac = string(correct_faces, "/", num_faces)
    faces_percent = string("(", correct_faces_percent, "% of faces were recognised as faces)")
    non_faces_frac = string(correct_non_faces, "/", num_non_faces)
    non_faces_percent = string("(", correct_non_faces_percent, "% of non-faces were identified as non-faces)")

    @info("...done.\n")
    @info("Result:\n")

    @printf("%10.9s %10.15s %15s\n", "Faces:", faces_frac, faces_percent)
    @printf("%10.9s %10.15s %15s\n\n", "Non-faces:", non_faces_frac, non_faces_percent)
    =#
    face_testing_image_paths       = readdir(joinpath(testing_path, "Faces"), join=true, sort=false)
    pareidolia_testing_image_paths = readdir(joinpath(testing_path, "Pareidolia"), join=true, sort=false)
    flower_testing_image_paths     = readdir(joinpath(testing_path, "Flowers"), join=true, sort=false)
    object_testing_image_paths     = readdir(joinpath(testing_path, "Objects"), join=true, sort=false)
    
    face_testing_images       = load_image.(face_testing_image_paths, scale=scale, scale_to=scale_to)
    pareidolia_testing_images = load_image.(pareidolia_testing_image_paths, scale=scale, scale_to=scale_to)
    flower_testing_images     = load_image.(flower_testing_image_paths, scale=scale, scale_to=scale_to)
    object_testing_images     = load_image.(object_testing_image_paths, scale=scale, scale_to=scale_to)
    
	num_faces      = length(face_testing_images)
    num_pareidolia = length(pareidolia_testing_images)
    num_flowers    = length(flower_testing_images)
    num_objects    = length(object_testing_images)
    
    face_scores       = Float64[ensemble_vote(img, classifiers) for img in face_testing_images]
    pareidolia_scores = Float64[ensemble_vote(img, classifiers) for img in pareidolia_testing_images]
    flower_scores     = Float64[ensemble_vote(img, classifiers) for img in flower_testing_images]
    object_scores     = Float64[ensemble_vote(img, classifiers) for img in object_testing_images]
    
    face_faceness_scores       = Float64[get_faceness(classifiers, img) for img in face_testing_images]
    pareidolia_faceness_scores = Float64[get_faceness(classifiers, img) for img in pareidolia_testing_images]
    flower_faceness_scores     = Float64[get_faceness(classifiers, img) for img in flower_testing_images]
    object_faceness_scores     = Float64[get_faceness(classifiers, img) for img in object_testing_images]
	
    face_names       = String[basename(img) for img in face_testing_image_paths]
    pareidolia_names = String[basename(img) for img in pareidolia_testing_image_paths]
    flower_names     = String[basename(img) for img in flower_testing_image_paths]
    object_names     = String[basename(img) for img in object_testing_image_paths]
    
    scores_df = DataFrame(image_name = String[], image_type = Int[], image_score_binary = Int8[], faceness_score = Float64[])
    anova_df = DataFrame(image_type = Int[],  faceness_score = Float64[])
    for (image_type, image_names, image_scores, faceness_scores) in ((FACE, face_names, face_scores, face_faceness_scores), (PAREIDOLIA, pareidolia_names, pareidolia_scores, pareidolia_faceness_scores), (FLOWER, flower_names, flower_scores, flower_faceness_scores), (OBJECT, object_names, object_scores, object_faceness_scores))
        for (n, s, f) in zip(image_names, image_scores, faceness_scores)
            push!(scores_df, (n, Int(image_type), s, f))
            push!(anova_df, (Int(image_type), f))
        end
    end
    
    # write score data
	data_file = joinpath(dirname(dirname(@__DIR__)), "data", "faceness-scores.csv")
    write(data_file, scores_df, writeheader=true)
    @info("...done.  Dataset written to $(data_file).\n")
    
    ### FACES VS OBJECTS
    
    @info("Computing differences in scores between faces and objects...")
    welch_t = UnequalVarianceTTest(face_faceness_scores, object_faceness_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")
    
    gr() # set plot backend
    theme(:solarized)
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(face_faceness_scores, xaxis=false, label = false),
                    StatsPlots.boxplot(object_faceness_scores, xaxis=false, label = false),
                    title = ["Facenesses of Faces" "Facenesses of Objects"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    link = :y,
                )
    StatsPlots.savefig(plot, joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_objects.pdf"))
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_objects.pdf"))")
    
    ### PAREIDOLIA VS OBJECTS
    
    @info("Computing differences in scores between pareidolia and objects...")
    welch_t = UnequalVarianceTTest(pareidolia_faceness_scores, object_faceness_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")
    
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(pareidolia_faceness_scores, xaxis=false, label = false),
                    StatsPlots.boxplot(object_faceness_scores, xaxis=false, label = false),
                    title = ["Facenesses of Pareidolia" "Facenesses of Objects"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    link = :y,
                )
    StatsPlots.savefig(plot, joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_objects.pdf"))
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_objects.pdf"))")
    
    ### FACES VS FLOWERS
    
    @info("Computing differences in scores between faces and flowers...")
    welch_t = UnequalVarianceTTest(face_faceness_scores, flower_faceness_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")
    
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(face_faceness_scores, xaxis=false, label = false),
                    StatsPlots.boxplot(flower_faceness_scores, xaxis=false, label = false),
                    title = ["Facenesses of Faces" "Facenesses of Flowers"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    link = :y,
                )
    StatsPlots.savefig(plot, joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_flowers.pdf"))
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_faces_versus_flowers.pdf"))")
    
    ### PAREIDOLIA VS FLOWERS
    
    @info("Computing differences in scores between pareidolia and flowers...")
    welch_t = UnequalVarianceTTest(pareidolia_faceness_scores, flower_faceness_scores)
    @info("...done.  $welch_t\n")
    @info("Constructing box plot with said dataset...")
    
    plot = StatsPlots.plot(
                    StatsPlots.boxplot(pareidolia_faceness_scores, xaxis=false, label = false),
                    StatsPlots.boxplot(flower_faceness_scores, xaxis=false, label = false),
                    title = ["Facenesses of Pareidolia" "Facenesses of Flowers"],
                    fontfamily = font("Times"),
                    layout = @layout([a b]),
                    link = :y,
                )
    StatsPlots.savefig(plot, joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_flowers.pdf"))
    @info("...done.  Plot created at $(joinpath(dirname(dirname(@__DIR__)), "figs", "faceness_of_pareidolia_versus_flowers.pdf"))")
end

# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_2000_pos_2000_neg_(128,128)_(100,100,30,30)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_5000_pos_5000_neg_(128,128)_(100,100,30,30)")
data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_1500_pos_1500_neg_(128,128)_(128,128,1,1)")
# data_file = joinpath(dirname(@__DIR__), "data", "classifiers_10_from_4000_pos_4000_neg_(24,24)_(-1,-1,1,1)")
@time main(data_file, 500, 500, scale=true, scale_to=(128, 128))
