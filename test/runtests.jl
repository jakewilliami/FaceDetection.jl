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
# @test

