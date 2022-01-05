#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=512G
#SBATCH --partition=bigmem
#SBATCH --account=psyc
#SBATCH --constraint=AVX
#SBATCH --time=05-00:00:00
#SBATCH --job-name=FaceDetection.jl
#SBATCH -o /nfs/home/collyeli/facedetection.jl.out
#SBATCH -e /nfs/home/collyeli/facedetection.jl.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakewilliami@icloud.com

module load julia
julia -tauto --project=/nfs/home/collyeli/projects/FaceDetection.jl/ /nfs/home/collyeli/projects/FaceDetection.jl/examples/write_ffhq_things.jl
