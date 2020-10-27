#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

# we assume that `smart_choose_feats = true`
main_path = dirname(dirname(@__FILE__))
data_path = joinpath(main_path, "data")
main_image_path = joinpath(main_path, "data", "main")
alt_image_path = joinpath(main_path, "data", "alt")

num_classifiers = 10
