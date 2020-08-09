#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/src/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#
    
