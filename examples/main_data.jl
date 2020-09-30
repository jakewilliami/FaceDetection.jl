#!/usr/bin/env bash
    #=
    exec julia --project="$(realpath $(dirname $0))/" "${BASH_SOURCE[0]}" "$@" -e "include(popfirst!(ARGS))" \
    "${BASH_SOURCE[0]}" "$@"
    =#

# pos_training_path = joinpath(main_image_path, "trainset", "faces")
# neg_training_path = joinpath(main_image_path, "trainset", "non-faces")
# pos_testing_path = joinpath(main_image_path, "testset", "faces")#joinpath(homedir(), "Desktop", "faces")#"$main_image_path/testset/faces/"
# neg_testing_path = joinpath(main_image_path, "testset", "non-faces")

pos_training_path = joinpath(data_path, "lfw-all")
neg_training_path = joinpath(data_path, "all-non-faces")
pos_testing_path = joinpath(data_path, "lizzie-testset", "faces")
neg_testing_path = joinpath(data_path, "lizzie-testset", "nonfaces")
