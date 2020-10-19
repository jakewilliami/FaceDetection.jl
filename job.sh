#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --mem=1000G
#SBATCH --partition=bigmem
#SBATCH --time=3-00:00:00
#SBATCH -o /nfs/home/irelanjake/fd-project1.out
#SBATCH -e /nfs/home/irelanjake/fd-project1.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakewilliami@icloud.com
#SBATCH --account=psyc
#SBATCH --constraint=AVX

EXAMPLE_DIR="${HOME}/projects/FaceDetection.jl/examples/"

sed 's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 2/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 3/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 4/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 5/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 6/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 7/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 8/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 9/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

sed	's/^num_classifiers[[:space:]]=[[:space:]].*$/num_classifiers = 10/g' "${EXAMPLE_DIR}/constants.jl"
${EXAMPLE_DIR}/write.jl

