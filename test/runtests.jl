#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(realpath $(dirname $0))))/examples/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl")) # ../src/FaceDetection.jl

using .FaceDetection
using Test: @testset, @test
using Suppressor: @suppress
# using BenchmarkTools: @btime

const main_data_path = joinpath(@__DIR__, "images")

@time @testset "FaceDetection.jl" begin
    # IntegralImage.jl
    @test isequal(FaceDetection.to_integral_image([17 24 1 8 15; 23 5 7 14 16; 4 6 13 20 22; 10 12 19 21 3; 11 18 25 2 9]), [17 41 42 50 65; 40 69 77 99 130; 44 79 100 142 195; 54 101 141 204 260; 65 130 195 260 325])
    @test isequal(FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)), 18)
    @test typeof(FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5))) <: AbstractFloat
    @test FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)) isa AbstractFloat

    # HaarLikeFeature.jl
    a = tuple(rand(Int), rand(Int))
    b = tuple(rand(Int), rand(Int))
    c = rand(Int)
    d = rand(Int)
    e = rand((0, 1))
    f = rand((0, 1))
    arr = rand(Int, 100, 100)
    @test FaceDetection.HaarLikeObject(a, b, c, d, e, f) isa HaarLikeObject
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).feature_type isa Tuple{Integer, Integer}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).position isa Tuple{Integer, Integer}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).top_left isa Tuple{Integer, Integer}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).bottom_right isa Tuple{Integer, Integer}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).width isa Integer
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).height isa Integer
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).threshold ∈ [0, 1]
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).polarity ∈ [0, 1]
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).weight ∈ [0, 1]
    @test FaceDetection.get_vote(FaceDetection.HaarLikeObject(a, b, c, d, e, f), arr) ∈ [-1, 1]

    # AdaBoost.jl
	pos_training_path, neg_training_path, pos_testing_path, neg_testing_path = string(), string(), string(), string()
	classifiers = []
	features = []
	@suppress begin
		pos_training_path = joinpath(main_data_path, "pos")
	    neg_training_path = joinpath(main_data_path, "neg")
	    classifiers = learn(pos_training_path, neg_training_path, 10, 8, 10, 8, 10)
		features = FaceDetection.create_features(19, 19, 8, 10, 8, 10)
	end
	@test length(features) == 4520
	
    # Utils.jl
	p, n = 0, 0
	@suppress begin
		pos_testing_path = joinpath(main_data_path, "pos_testing")
	    neg_testing_path = joinpath(main_data_path, "neg_testing")
		num_faces = length(filtered_ls(pos_testing_path))
		num_non_faces = length(filtered_ls(neg_testing_path))
		p = sum(ensemble_vote_all(pos_testing_path, classifiers)) / num_faces
		n = (num_non_faces - sum(ensemble_vote_all(neg_testing_path, classifiers))) / num_non_faces
	end
	@test isapprox(p, 0.63, atol=1e-1)
	@test isapprox(n, 0.372, atol=1e-1)
	random_img = load_image(rand(vcat(filtered_ls.([pos_training_path, neg_training_path, pos_testing_path, neg_testing_path])...)))
	@test get_faceness(classifiers[rand(1:length(classifiers))], random_img) isa Real
	@test determine_feature_size(pos_training_path, neg_training_path) == (10, 10, 8, 8, (19, 19))
end # end tests
