#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $(dirname $0)))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

include(joinpath(dirname(dirname(@__FILE__)), "src", "FaceDetection.jl")) # ../src/FaceDetection.jl

using .FaceDetection
using Test

@testset "FaceDetection.jl" begin
    # IntegralImage.jl
    @test isequal(FaceDetection.to_integral_image([17 24 1 8 15; 23 5 7 14 16; 4 6 13 20 22; 10 12 19 21 3; 11 18 25 2 9]), [17 41 42 50 65; 40 69 77 99 130; 44 79 100 142 195; 54 101 141 204 260; 65 130 195 260 325])
    @test isequal(FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)), 18)
    @test typeof(FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5))) <: AbstractFloat
    @test FaceDetection.sum_region(FaceDetection.to_integral_image([1 7 4 2 9.9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)) isa AbstractFloat

    # HaarLikeFeature.jl
    a = (rand(Int), rand(Int))
    b = (rand(Int), rand(Int))
    c = rand(Int)
    d = rand(Int)
    e = rand((0, 1))
    f = rand((0, 1))
    arr = rand(Int, 100, 100)
    @test FaceDetection.HaarLikeObject(a, b, c, d, e, f) isa HaarLikeObject
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).feature_type isa Tuple{Int, Int}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).position isa Tuple{Int, Int}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).top_left isa Tuple{Int, Int}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).bottom_right isa Tuple{Int, Int}
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).width isa Int
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).height isa Int
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).threshold ∈ [0, 1]
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).polarity ∈ [0, 1]
    @test FaceDetection.HaarLikeObject((1,3), (1,3), 10, 8, 0, 1).weight ∈ [0, 1]
    @test FaceDetection.get_vote(FaceDetection.HaarLikeObject(a, b, c, d, e, f), arr) ∈ [-1, 1]

    # AdaBoost.jl
    # @test

    # Utils.jl
    # @test
end # end tests
