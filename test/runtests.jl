#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

include(joinpath(homedir(), "FaceDetection.jl", "src", "FaceDetection.jl"))

using .FaceDetection
using Test: @test

# write your own tests here
@test 1 == 1
@test sumRegion(toIntegralImage([1 7 4 2 9; 7 2 3 8 2; 1 8 7 9 1; 3 2 3 1 5; 2 9 5 6 6]), (4,4), (5,5)) == 18
