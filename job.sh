#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=parallel
#SBATCH --time=3-12:00
#SBATCH -o /nfs/home/irelanjake/project1.out
#SBATCH -e /nfs/home/irelanjake/project1.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakewilliami@icloud.com

EXAMPLE_DIR="${HOME}/projects/FaceDetection.jl/examples/"

sed 's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 2/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 3/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 4/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 5/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 6/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 7/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 8/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 9/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 10/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 20/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 50/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 100/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 200/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 1000/g' "${EXAMPLE_DIR}/constants.jl"
julia "${EXAMPLE_DIR}/write.jl"
