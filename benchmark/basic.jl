Threads.nthreads() == 1 && @warn("You are currently only using one thread, when the programme supports multithreading")

using FaceDetection
using Printf: @printf
using Images: imresize
using Logging
using BenchmarkTools: @btime

Logging.disable_logging(Logging.Info)

include(joinpath(dirname(@__DIR__), "examples", "constants.jl"))
include(joinpath(dirname(@__DIR__), "examples", "main_data.jl"))

min_size_img = (19, 19)
scale, scale_to = false, (200, 200)

min_feature_height = 8
max_feature_height = 10
min_feature_width = 8
max_feature_width = 10

learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = scale, scale_to = scale_to, show_progress = false)

@btime learn(pos_training_path, neg_training_path, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width; scale = scale, scale_to = scale_to, show_progress = false)
