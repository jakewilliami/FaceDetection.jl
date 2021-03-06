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

@time @testset "FaceDetection.jl" begin
	# constants and variables
	main_data_path = joinpath(@__DIR__, "images")
	pos_training_path = joinpath(main_data_path, "pos")
	neg_training_path = joinpath(main_data_path, "neg")
	pos_testing_path = joinpath(main_data_path, "pos_testing")
	neg_testing_path = joinpath(main_data_path, "neg_testing")
	a, b, c, d, e, f = tuple(rand(Int), rand(Int)), tuple(rand(Int), rand(Int)), rand(Int), rand(Int), rand((0, 1)), rand((0, 1))
    arr = rand(Int, 100, 100)
	int_img = load_image(rand(vcat(filtered_ls.([pos_training_path, neg_training_path, pos_testing_path, neg_testing_path])...)), scale = true, scale_to = (24, 24))
	feature_2v = HaarLikeObject(feature_types.two_vertical, (1, 1), 10, 10, 100000, 1)
	feature_2h = HaarLikeObject(feature_types.two_horizontal, (1, 1), 10, 10, 100000, 1)
	feature_3h = HaarLikeObject(feature_types.three_horizontal, (1, 1), 10, 10, 100000, 1)
	feature_3v = HaarLikeObject(feature_types.two_vertical, (1, 1), 10, 10, 100000, 1)
	feature_4 = HaarLikeObject(feature_types.four, (1, 1), 10, 10, 100000, 1)
	left_area = sum_region(int_img, (1, 1), (24, 12))
	right_area = sum_region(int_img, (1, 12), (24, 24))
	left_area_3h = sum_region(int_img, (1, 1), (8, 24))
	middle_area_3h = sum_region(int_img, (8, 1), (16, 24))
	right_area_3h = sum_region(int_img, (16, 1), (24, 24))
	left_area_3v = sum_region(int_img, (1, 1), (24, 8))
	middle_area_3v = sum_region(int_img, (1, 8), (24, 16))
	right_area_3v = sum_region(int_img, (1, 16), (24, 24))
	top_left_area = sum_region(int_img, (1, 1), (12, 12))
	top_right_area = sum_region(int_img, (12, 1), (24, 12))
	bottom_left_area = sum_region(int_img, (1, 12), (12, 24))
	bottom_right_area = sum_region(int_img, (12, 12), (24, 24))
	expected_2v = feature_2v.threshold * feature_2v.polarity > left_area - right_area ? 1 : 0
	expected_2v_fail = feature_2h.threshold * -1 > left_area - right_area ? 1 : 0
	expected_2h = feature_2h.threshold * feature_2h.polarity > left_area - right_area ? 1 : 0
	expected_3h = feature_3h.threshold * feature_3h.polarity > left_area_3h - middle_area_3h + right_area_3h ? 1 : 0
	expected_3v = feature_3v.threshold * feature_3v.polarity > left_area_3v - middle_area_3v + right_area_3v ? 1 : 0
	expected_4 = feature_4.threshold * feature_4.polarity > top_left_area - top_right_area - bottom_left_area + bottom_right_area ? 1 : 0
	classifiers = []
	features = []
	p, n = 0, 0
	random_img = load_image(rand(vcat(filtered_ls.([pos_training_path, neg_training_path, pos_testing_path, neg_testing_path])...)))
	
    # IntegralImage.jl
    @test isequal(to_integral_image([17 24 1 8 15; 23 5 7 14 16; 4 6 13 20 22; 10 12 19 21 3; 11 18 25 2 9]), [17 41 42 50 65; 40 69 77 99 130; 44 79 100 142 195; 54 101 141 204 260; 65 130 195 260 325])
    @test isequal(sum_region(to_integral_image([1 7 4 2 9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)), 18)
    @test typeof(sum_region(to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5))) <: AbstractFloat
    @test sum_region(to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)) isa AbstractFloat

    # HaarLikeFeature.jl
    @test HaarLikeObject(a, b, c, d, e, f) isa HaarLikeObject
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).feature_type isa Tuple{Integer, Integer}
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).position isa Tuple{Integer, Integer}
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).top_left isa Tuple{Integer, Integer}
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).bottom_right isa Tuple{Integer, Integer}
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).width isa Integer
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).height isa Integer
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).threshold ∈ [0, 1]
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).polarity ∈ [0, 1]
    @test HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).weight ∈ [0, 1]
    @test get_vote(HaarLikeObject(a, b, c, d, e, f), arr) ∈ [-1, 1]
	@test get_vote(feature_2v, int_img) == expected_2v
	@test get_vote(feature_2v, int_img) != expected_2v_fail
	@test get_vote(feature_2h, int_img) == expected_2h
	@test get_vote(feature_3h, int_img) == expected_3h
	@test get_vote(feature_3v, int_img) == expected_3v
	@test get_vote(feature_4, int_img) == expected_4

    # AdaBoost.jl
	@suppress begin
	    classifiers = learn(pos_training_path, neg_training_path, 10, 8, 10, 8, 10)
		features = FaceDetection.create_features(19, 19, 8, 10, 8, 10)
	end
	@test length(features) == 4520
	
    # Utils.jl
	@test determine_feature_size(pos_training_path, neg_training_path) == (10, 10, 8, 8, (19, 19))
	@test get_faceness(classifiers[rand(1:length(classifiers))], random_img) isa Real
	@suppress begin
		num_faces = length(filtered_ls(pos_testing_path))
		num_non_faces = length(filtered_ls(neg_testing_path))
		p = sum(ensemble_vote_all(pos_testing_path, classifiers)) / num_faces
		n = (num_non_faces - sum(ensemble_vote_all(neg_testing_path, classifiers))) / num_non_faces
	end
	@test isapprox(p, 0.496, atol=1e-1) # these tests implicitly test the whole algorithm
	@test isapprox(n, 0.536, atol=1e-1)
end # end tests
